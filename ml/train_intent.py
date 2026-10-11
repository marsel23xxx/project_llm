"""Train the intent classifier that routes questions without an LLM call.

Compares two scikit-learn pipelines, picks the best on held-out templates, reports accuracy on
hand-written questions it never saw, and exports it to backend/models/intent.joblib.

Usage: python intent/generate.py && python train_intent.py
"""

import time

import joblib
import mlflow
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from common import BACKEND_MODELS, ROOT, write_report

from app.ml.text import normalize_batch  # noqa: E402  (path set up in common)

SEED = 42
# The router trusts the classifier only above a confidence threshold, chosen on validation templates
# as the lowest one whose decisions are at least this accurate; the rest go to the LLM.
TARGET_PRECISION = 0.97
THRESHOLDS = [round(0.4 + 0.05 * i, 2) for i in range(12)]


def pick_threshold(model: Pipeline, texts: pd.Series, labels: pd.Series) -> tuple[float, dict]:
    probs = model.predict_proba(texts)
    pred = model.classes_[probs.argmax(axis=1)]
    conf = probs.max(axis=1)
    for t in THRESHOLDS:
        mask = conf >= t
        if mask.sum() and (pred[mask] == labels.to_numpy()[mask]).mean() >= TARGET_PRECISION:
            return t, {"threshold": t, "val_precision": round(float((pred[mask] == labels.to_numpy()[mask]).mean()), 4), "val_coverage": round(float(mask.mean()), 4)}
    return THRESHOLDS[-1], {"threshold": THRESHOLDS[-1], "note": "target precision not reached"}


def routed(model: Pipeline, texts: pd.Series, labels: pd.Series, threshold: float) -> dict:
    probs = model.predict_proba(texts)
    pred = model.classes_[probs.argmax(axis=1)]
    mask = probs.max(axis=1) >= threshold
    return {
        "coverage": round(float(mask.mean()), 4),
        "precision_when_trusted": round(float((pred[mask] == labels.to_numpy()[mask]).mean()), 4) if mask.any() else None,
    }


def features() -> FeatureUnion:
    # Words catch "korelasi"/"sentimen"; character n-grams survive typos and slang ("korlasi", "brp").
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
        ]
    )


def candidates() -> dict[str, Pipeline]:
    clean = FunctionTransformer(normalize_batch)
    return {
        "tfidf_logreg": Pipeline(
            [("clean", clean), ("features", features()), ("clf", LogisticRegression(C=10, max_iter=2000))]
        ),
        "tfidf_linearsvc": Pipeline(
            [
                ("clean", clean),
                ("features", features()),
                # Calibrated so the router gets a usable confidence score.
                ("clf", CalibratedClassifierCV(LinearSVC(C=0.5), cv=3)),
            ]
        ),
    }


def evaluate(model: Pipeline, texts: pd.Series, labels: pd.Series) -> dict:
    start = time.perf_counter()
    pred = model.predict(texts)
    ms_per_item = (time.perf_counter() - start) * 1000 / len(texts)
    return {
        "accuracy": round(accuracy_score(labels, pred), 4),
        "macro_f1": round(f1_score(labels, pred, average="macro"), 4),
        "ms_per_question": round(ms_per_item, 3),
    }


def main() -> None:
    data = pd.read_csv(ROOT / "data" / "intent" / "train.csv")
    heldout = pd.read_csv(ROOT / "intent" / "heldout.csv")
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(split.split(data, groups=data["group"]))
    train, val = data.iloc[train_idx], data.iloc[val_idx]

    mlflow.set_experiment("intent-classifier")
    results = {}
    for name, model in candidates().items():
        with mlflow.start_run(run_name=name):
            model.fit(train["text"], train["label"])
            val_metrics = evaluate(model, val["text"], val["label"])
            heldout_metrics = evaluate(model, heldout["text"], heldout["label"])
            mlflow.log_params({"model": name, "train_rows": len(train), "sklearn": sklearn.__version__})
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            mlflow.log_metrics({f"heldout_{k}": v for k, v in heldout_metrics.items()})
            results[name] = {"model": model, "val": val_metrics, "heldout": heldout_metrics}
            print(f"{name:18s} val={val_metrics}  heldout={heldout_metrics}")

    # Model selection uses validation only; the hand-written set is reported, never optimised on.
    best_name = max(results, key=lambda n: results[n]["val"]["macro_f1"])
    threshold, threshold_info = pick_threshold(results[best_name]["model"], val["text"], val["label"])

    # Refit on everything (train + held-out templates) before export; the hand-written set stays unseen.
    final = candidates()[best_name].fit(data["text"], data["label"])
    final_heldout = evaluate(final, heldout["text"], heldout["label"])
    final_heldout["routing"] = routed(final, heldout["text"], heldout["label"], threshold)
    report = classification_report(heldout["label"], final.predict(heldout["text"]), output_dict=True, zero_division=0)
    errors = [
        {"text": t, "label": y, "predicted": p}
        for t, y, p in zip(heldout["text"], heldout["label"], final.predict(heldout["text"]))
        if y != p
    ]

    BACKEND_MODELS.mkdir(parents=True, exist_ok=True)
    artifact = BACKEND_MODELS / "intent.joblib"
    joblib.dump(
        {
            "pipeline": final,
            "labels": list(final.classes_),
            "model": best_name,
            "sklearn_version": sklearn.__version__,
            "threshold": threshold,
            "heldout": final_heldout,
        },
        artifact,
        compress=3,
    )
    path = write_report(
        "intent",
        {
            "selected": best_name,
            "candidates": {n: {"val": r["val"], "heldout": r["heldout"]} for n, r in results.items()},
            "threshold": threshold_info,
            "final_heldout": final_heldout,
            "per_class": {k: v for k, v in report.items() if k in final.classes_},
            "heldout_errors": errors,
            "artifact": str(artifact.relative_to(ROOT.parent)),
            "artifact_kb": round(artifact.stat().st_size / 1024, 1),
        },
    )
    print(f"\nTerpilih: {best_name}. Held-out manual: {final_heldout}")
    print(f"Salah klasifikasi ({len(errors)}):")
    for e in errors:
        print(f"  [{e['label']} -> {e['predicted']}] {e['text']}")
    print(f"Model: {artifact} ({artifact.stat().st_size / 1024:.0f} KB), laporan: {path}")


if __name__ == "__main__":
    main()
