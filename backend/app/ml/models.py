"""Load the trained ML models that ship with the backend (backend/models/).

- intent.joblib              scikit-learn intent router (ml/train_intent.py)
- sentiment_baseline.joblib  scikit-learn SmSA sentiment model (ml/train_sentiment_baseline.py)
- sentiment_onnx/            optional IndoBERT ONNX export (ml/colab); preferred when present

Every model is optional: a missing file just means the agent falls back to the LLM.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

logger = logging.getLogger(__name__)

SENTIMENT_LABELS_ID = {"positive": "positif", "neutral": "netral", "negative": "negatif"}


@dataclass
class Prediction:
    label: str
    confidence: float


class IntentClassifier:
    def __init__(self, path: Path):
        bundle = joblib.load(path)
        self.pipeline = bundle["pipeline"]
        self.labels: list[str] = bundle["labels"]
        self.name = bundle["model"]
        # Chosen during training on validation data (ml/train_intent.py).
        self.threshold: float = bundle.get("threshold", 0.6)

    def predict(self, text: str) -> Prediction:
        probs = self.pipeline.predict_proba([text])[0]
        best = int(np.argmax(probs))
        return Prediction(self.labels[best], float(probs[best]))


class SklearnSentiment:
    # Sub-millisecond per text, so a whole column fits in one request.
    max_texts = 5000

    def __init__(self, path: Path):
        bundle = joblib.load(path)
        self.pipeline = bundle["pipeline"]
        self.labels: list[str] = bundle["labels"]
        self.name = f"sklearn:{bundle['model']}"

    def predict(self, texts: list[str]) -> list[Prediction]:
        probs = self.pipeline.predict_proba(texts)
        best = probs.argmax(axis=1)
        return [Prediction(self.labels[i], float(p[i])) for i, p in zip(best, probs)]


class OnnxSentiment:
    """IndoBERT fine-tuned on SmSA, exported to ONNX and int8-quantized (see ml/colab)."""

    BATCH = 32
    # ~90 ms per text on CPU: analyse a random sample so the answer arrives in about half a minute.
    max_texts = 300

    def __init__(self, directory: Path):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        config = joblib.load(directory / "config.joblib")
        self.labels: list[str] = config["labels"]
        self.max_length: int = config["max_length"]
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.enable_truncation(self.max_length)
        self.tokenizer.enable_padding(pad_id=config["pad_id"], pad_token=config["pad_token"])
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(directory / "model.int8.onnx"), options, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.name = "onnx:indobert-int8"

    def predict(self, texts: list[str]) -> list[Prediction]:
        out: list[Prediction] = []
        for start in range(0, len(texts), self.BATCH):
            encodings = self.tokenizer.encode_batch([str(t) for t in texts[start : start + self.BATCH]])
            feed = {
                "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            }
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)
            logits = self.session.run(None, {k: v for k, v in feed.items() if k in self.input_names})[0]
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp / exp.sum(axis=1, keepdims=True)
            out.extend(Prediction(self.labels[i], float(p[i])) for i, p in zip(probs.argmax(axis=1), probs))
        return out


@dataclass
class MLModels:
    intent: IntentClassifier | None
    sentiment: SklearnSentiment | OnnxSentiment | None


def load_models(directory: Path) -> MLModels:
    intent = sentiment = None
    try:
        intent = IntentClassifier(directory / "intent.joblib")
    except FileNotFoundError:
        logger.warning("intent.joblib tidak ditemukan; routing memakai LLM")

    onnx_dir = directory / "sentiment_onnx"
    if (onnx_dir / "model.int8.onnx").exists():
        try:
            sentiment = OnnxSentiment(onnx_dir)
        except Exception:
            logger.exception("Gagal memuat model ONNX; memakai baseline scikit-learn")
    if sentiment is None:
        try:
            sentiment = SklearnSentiment(directory / "sentiment_baseline.joblib")
        except FileNotFoundError:
            logger.warning("Model sentimen tidak ditemukan; analisis sentimen tidak tersedia")

    logger.info(
        "Model ML: intent=%s, sentimen=%s",
        intent.name if intent else "-",
        sentiment.name if sentiment else "-",
    )
    return MLModels(intent=intent, sentiment=sentiment)
