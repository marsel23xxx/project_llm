from pathlib import Path

import pandas as pd
import pytest

from app.core.config import Settings
from app.ml.models import MLModels, Prediction, load_models
from app.ml.text import normalize
from app.services.agent.graph import build_agent, stream_agent
from app.services.agent.sentiment import SentimentUnavailable, analyze, pick_columns
from app.services.data.ingest import write_duckdb

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


class KeywordSentiment:
    name = "fake"
    max_texts = 5000

    def predict(self, texts):
        out = []
        for t in texts:
            label = "negative" if "buruk" in t else "positive" if "enak" in t else "neutral"
            out.append(Prediction(label, 0.9))
        return out


class FixedIntent:
    name = "fake-intent"
    threshold = 0.6

    def __init__(self, label, confidence):
        self.label, self.confidence = label, confidence

    def predict(self, text):
        return Prediction(self.label, self.confidence)


class ScriptedLLM:
    def __init__(self, replies, strong=False):
        self.replies = list(replies)
        self.has_strong_model = strong

    async def complete(self, kind, system, messages, max_tokens=4096):
        from app.services.llm.base import LLMResult

        return LLMResult(self.replies.pop(0), "fake", "fake", 1, 1)


class SummaryLLM:
    def __init__(self):
        self.calls = []

    async def complete(self, kind, system, messages, max_tokens=4096):
        from app.services.llm.base import LLMResult

        self.calls.append(messages[-1]["content"])
        return LLMResult("Mayoritas ulasan positif.", "fake", "fake", 1, 1)


@pytest.fixture
def reviews_db(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "cabang": ["Bandung", "Bandung", "Medan", "Medan", "Medan"],
            "ulasan": [
                "makanannya enak sekali dan murah",
                "pelayanan buruk sekali, lama banget",
                "rasanya enak, tempatnya nyaman",
                "biasa saja tidak ada yang spesial",
                "kopinya enak dan baristanya ramah",
            ],
            "rating": [5, 1, 5, 3, 5],
        }
    )
    path = tmp_path / "ulasan.duckdb"
    write_duckdb(df, path)
    return path


def test_normalize_handles_slang_and_noise():
    assert normalize("Brp omzet yg bulan lalu?? https://x.co @admin") == "berapa omzet yang bulan lalu"
    assert normalize("bagussss bgt gk ada lawan") == "baguss banget tidak ada lawan"


def test_pick_columns_finds_text_and_named_group(reviews_db):
    assert pick_columns(reviews_db, "analisis sentimen ulasan") == ("ulasan", None)
    assert pick_columns(reviews_db, "sentimen ulasan per cabang") == ("ulasan", "cabang")


def test_analyze_counts_labels_and_groups(reviews_db):
    result, meta = analyze(reviews_db, "sentimen per cabang", KeywordSentiment())
    assert result.columns[0] == "cabang"
    assert result.rows[0][0] == "Bandung"  # highest negative share first
    assert result.rows[0][1:4] == [1, 0, 1]
    assert meta["analyzed"] == 5 and meta["samples"][0]["label"] == "negatif"

    overall, _ = analyze(reviews_db, "bagaimana sentimen pelanggan", KeywordSentiment())
    assert overall.rows == [["positif", 3, 60.0], ["netral", 1, 20.0], ["negatif", 1, 20.0]]


def test_analyze_samples_when_model_is_slow(reviews_db):
    slow = KeywordSentiment()
    slow.max_texts = 3
    result, meta = analyze(reviews_db, "bagaimana sentimen pelanggan", slow)
    assert meta["analyzed"] == 3 and meta["total"] == 5 and meta["sampled"]
    assert sum(row[1] for row in result.rows) == 3


def test_analyze_rejects_dataset_without_text(tmp_path):
    path = tmp_path / "angka.duckdb"
    write_duckdb(pd.DataFrame({"kode": ["A", "B"], "nilai": [1, 2]}), path)
    with pytest.raises(SentimentUnavailable):
        analyze(path, "sentimen", KeywordSentiment())


async def test_confident_intent_model_skips_llm_routing(reviews_db):
    llm = SummaryLLM()
    agent = build_agent(llm, Settings(), MLModels(FixedIntent("sentiment", 0.95), KeywordSentiment()))
    initial = {"question": "sentimen ulasan", "context": "", "db_path": str(reviews_db), "usage": []}
    events = [e async for e, _ in stream_agent(agent, initial)]
    assert [e["type"] for e in events] == ["intent", "table", "stdout", "chart", "samples", "answer"]
    assert len(llm.calls) == 1  # only the summary used the LLM


async def test_unsure_intent_model_trusted_when_only_local_llm(reviews_db):
    llm = ScriptedLLM(["Halo!"])
    agent = build_agent(llm, Settings(), MLModels(FixedIntent("chat", 0.3), None))
    initial = {"question": "hmm", "context": "", "db_path": str(reviews_db), "usage": []}
    results = [pair async for pair in stream_agent(agent, initial)]
    assert results[0][0]["value"] == "chat" and results[0][1][0]["provider"] == "ml"


async def test_unsure_intent_model_falls_back_to_strong_llm(reviews_db):
    llm = ScriptedLLM(["chat", "Halo!"], strong=True)
    agent = build_agent(llm, Settings(), MLModels(FixedIntent("sql", 0.3), None))
    initial = {"question": "hmm", "context": "", "db_path": str(reviews_db), "usage": []}
    results = [pair async for pair in stream_agent(agent, initial)]
    assert [e["type"] for e, _ in results] == ["intent", "answer"]
    assert results[0][0]["value"] == "chat"  # the LLM's choice, not the unsure model's "sql"
    assert results[-1][1][0]["provider"] == "fake"


@pytest.mark.skipif(not (MODELS_DIR / "intent.joblib").exists(), reason="model belum dilatih")
def test_shipped_models_load_and_predict():
    models = load_models(MODELS_DIR)
    assert models.intent.predict("hitung korelasi harga dan qty").label == "python"
    # Contract with the router: a wrong label must come with low confidence, so the LLM decides.
    pred = models.intent.predict("produk apa yang paling laku")
    assert pred.label == "sql" or pred.confidence < models.intent.threshold
    labels = [p.label for p in models.sentiment.predict(["makanannya enak banget", "pelayanannya buruk dan lambat"])]
    assert labels == ["positive", "negative"]
