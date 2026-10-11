"""Train the CPU sentiment baseline on IndoNLU SmSA (positive / neutral / negative).

This is the model the backend serves until an IndoBERT ONNX export is available
(see colab/indobert_sentiment.ipynb), and the baseline IndoBERT is compared against.

The official SmSA test labels are masked, so the official validation split is used as the
test set and a stratified slice of train is used for model selection.

Usage: python train_sentiment_baseline.py
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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from common import BACKEND_MODELS, ROOT, write_report

from app.ml.text import normalize_batch  # noqa: E402

SEED = 42
DATA = ROOT / "data" / "smsa"


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.tsv", sep="\t", header=None, names=["text", "label"], quoting=3)


def word_features(**kw) -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, **kw)


def candidates() -> dict[str, Pipeline]:
    clean = FunctionTransformer(normalize_batch)
    both = lambda: FeatureUnion(  # noqa: E731
        [("word", word_features()), ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True))]
    )
    return {
        "tfidf_nb": Pipeline([("clean", clean), ("features", word_features()), ("clf", ComplementNB(alpha=0.3))]),
        "tfidf_logreg": Pipeline(
            [("clean", clean), ("features", both()), ("clf", LogisticRegression(C=5, max_iter=3000, class_weight="balanced"))]
        ),
        "tfidf_linearsvc": Pipeline(
            [
                ("clean", clean),
                ("features", both()),
                ("clf", CalibratedClassifierCV(LinearSVC(C=0.3, class_weight="balanced"), cv=3)),
            ]
        ),
    }


def evaluate(model: Pipeline, df: pd.DataFrame) -> dict:
    start = time.perf_counter()
    pred = model.predict(df["text"])
    return {
        "accuracy": round(accuracy_score(df["label"], pred), 4),
        "macro_f1": round(f1_score(df["label"], pred, average="macro"), 4),
        "ms_per_text": round((time.perf_counter() - start) * 1000 / len(df), 3),
    }


def main() -> None:
    full_train = load("train_preprocess")
    test = load("valid_preprocess")
    train, val = train_test_split(full_train, test_size=0.1, stratify=full_train["label"], random_state=SEED)

    mlflow.set_experiment("sentiment-baseline")
    results = {}
    for name, model in candidates().items():
        with mlflow.start_run(run_name=name):
            start = time.perf_counter()
            model.fit(train["text"], train["label"])
            fit_seconds = round(time.perf_counter() - start, 1)
            val_metrics = evaluate(model, val)
            mlflow.log_params({"model": name, "train_rows": len(train), "sklearn": sklearn.__version__})
            mlflow.log_metrics({**{f"val_{k}": v for k, v in val_metrics.items()}, "fit_seconds": fit_seconds})
            results[name] = {"val": val_metrics, "fit_seconds": fit_seconds}
            print(f"{name:16s} fit={fit_seconds}s val={val_metrics}")

    best_name = max(results, key=lambda n: results[n]["val"]["macro_f1"])
    final = candidates()[best_name].fit(full_train["text"], full_train["label"])
    test_metrics = evaluate(final, test)
    report = classification_report(test["label"], final.predict(test["text"]), output_dict=True)
    with mlflow.start_run(run_name=f"{best_name}-final"):
        mlflow.log_params({"model": best_name, "train_rows": len(full_train)})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    artifact = BACKEND_MODELS / "sentiment_baseline.joblib"
    BACKEND_MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": final,
            "labels": list(final.classes_),
            "model": best_name,
            "sklearn_version": sklearn.__version__,
            "test": test_metrics,
        },
        artifact,
        compress=3,
    )
    path = write_report(
        "sentiment_baseline",
        {
            "dataset": "IndoNLU SmSA (test = official validation split, 1.260 teks)",
            "selected": best_name,
            "candidates": results,
            "test": test_metrics,
            "per_class": {k: report[k] for k in final.classes_},
            "artifact": str(artifact.relative_to(ROOT.parent)),
            "artifact_kb": round(artifact.stat().st_size / 1024, 1),
        },
    )
    print(f"\nTerpilih: {best_name}. Test: {test_metrics}")
    print(f"Model: {artifact} ({artifact.stat().st_size / 1024:.0f} KB), laporan: {path}")


if __name__ == "__main__":
    main()
