"""Fine-tune IndoBERT on SmSA (PyTorch), export to ONNX, quantize to int8, and benchmark.

Run on a GPU (Google Colab / Kaggle): see indobert_sentiment.ipynb in this folder.
Output: sentiment_onnx/ (copy to backend/models/sentiment_onnx/) + indobert_report.json

    python train_indobert.py --data-dir data/smsa --out sentiment_onnx
"""

import argparse
import inspect
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "indobenchmark/indobert-base-p1"
LABELS = ["negative", "neutral", "positive"]
MAX_LENGTH = 128
SEED = 42


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer):
        self.encodings = tokenizer(list(df["text"]), truncation=True, max_length=MAX_LENGTH)
        self.labels = [LABELS.index(label) for label in df["label"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {**{k: v[i] for k, v in self.encodings.items()}, "labels": self.labels[i]}


def load(data_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(data_dir / f"{name}.tsv", sep="\t", header=None, names=["text", "label"], quoting=3)


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
    }


def onnx_predict(session, tokenizer, texts: list[str], batch: int = 32) -> tuple[np.ndarray, float]:
    names = {i.name for i in session.get_inputs()}
    preds, start = [], time.perf_counter()
    for i in range(0, len(texts), batch):
        enc = tokenizer(texts[i : i + batch], truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="np")
        feed = {k: v.astype(np.int64) for k, v in enc.items() if k in names}
        preds.append(session.run(None, feed)[0].argmax(axis=1))
    return np.concatenate(preds), (time.perf_counter() - start) * 1000 / len(texts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/smsa"))
    parser.add_argument("--out", type=Path, default=Path("sentiment_onnx"))
    parser.add_argument("--epochs", type=float, default=3)
    args = parser.parse_args()
    torch.manual_seed(SEED)

    full = load(args.data_dir, "train_preprocess")
    test = load(args.data_dir, "valid_preprocess")  # official test labels are masked
    train, val = train_test_split(full, test_size=0.1, stratify=full["label"], random_state=SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)}
    )
    steps_per_epoch = math.ceil(len(train) / 32)
    wanted = dict(
        output_dir="checkpoints",
        num_train_epochs=args.epochs,
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_steps=int(0.1 * steps_per_epoch * args.epochs),
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
    )
    # TrainingArguments renames/removes options between transformers releases; pass only what exists.
    supported = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in supported:
        wanted["evaluation_strategy"] = wanted.pop("eval_strategy")
    dropped = sorted(k for k in wanted if k not in supported)
    if dropped:
        print("Diabaikan (tidak didukung versi transformers ini):", dropped)
    training_args = TrainingArguments(**{k: v for k, v in wanted.items() if k in supported})
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TextDataset(train, tokenizer),
        eval_dataset=TextDataset(val, tokenizer),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=lambda p: metrics(p.label_ids, p.predictions.argmax(axis=1)),
    )
    start = time.perf_counter()
    trainer.train()
    train_minutes = round((time.perf_counter() - start) / 60, 1)

    y_test = [LABELS.index(l) for l in test["label"]]
    torch_pred = trainer.predict(TextDataset(test, tokenizer)).predictions.argmax(axis=1)
    pytorch_metrics = metrics(y_test, torch_pred)

    # ---- Export to ONNX (on CPU, so the graph matches what the backend runs) ----
    args.out.mkdir(parents=True, exist_ok=True)
    model = trainer.model.to("cpu").eval()
    sample = tokenizer(["contoh kalimat"], return_tensors="pt")
    inputs = tuple(sample[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in sample)
    input_names = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in sample]
    fp32_path = args.out / "model.onnx"
    export_kwargs = dict(
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes={**{n: {0: "batch", 1: "sequence"} for n in input_names}, "logits": {0: "batch"}},
        opset_version=17,
    )
    try:
        # The TorchScript-based exporter handles dynamic_axes directly.
        torch.onnx.export(model, inputs, str(fp32_path), dynamo=False, **export_kwargs)
    except TypeError:
        # PyTorch versions without the `dynamo` switch.
        torch.onnx.export(model, inputs, str(fp32_path), **export_kwargs)
    int8_path = args.out / "model.int8.onnx"
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)

    # Sanity check: the exported graph must agree with PyTorch.
    with torch.no_grad():
        torch_logits = model(**sample).logits.numpy()
    fp32_session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    onnx_logits = fp32_session.run(None, {k: sample[k].numpy() for k in input_names})[0]
    assert np.allclose(torch_logits, onnx_logits, atol=1e-3), "ONNX export does not match PyTorch"

    # ---- CPU benchmark: PyTorch fp32 vs ONNX fp32 vs ONNX int8 ----
    texts = list(test["text"])
    torch.set_num_threads(2)
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tokenizer(texts[i : i + 32], truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
            model(**enc)
    torch_cpu_ms = (time.perf_counter() - start) * 1000 / len(texts)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    results = {}
    for name, path in (("onnx_fp32", fp32_path), ("onnx_int8", int8_path)):
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        pred, ms = onnx_predict(session, tokenizer, texts)
        results[name] = {**metrics(y_test, pred), "ms_per_text_cpu2": round(ms, 2), "size_mb": round(path.stat().st_size / 1e6, 1)}

    # Files the backend needs; the fp32 model is only an intermediate.
    fp32_path.unlink()
    tokenizer.backend_tokenizer.save(str(args.out / "tokenizer.json"))
    joblib.dump(
        {"labels": LABELS, "max_length": MAX_LENGTH, "pad_id": tokenizer.pad_token_id, "pad_token": tokenizer.pad_token},
        args.out / "config.joblib",
    )
    report = {
        "model": MODEL_NAME,
        "train_minutes": train_minutes,
        "test_rows": len(test),
        "pytorch_fp32": {**pytorch_metrics, "ms_per_text_cpu2": round(torch_cpu_ms, 2)},
        **results,
    }
    Path("indobert_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
