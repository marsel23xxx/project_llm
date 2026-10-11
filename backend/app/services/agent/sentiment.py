"""Sentiment analysis over a text column of the user's dataset, using the trained model."""

from collections import Counter
from pathlib import Path

import duckdb

from app.ml.models import SENTIMENT_LABELS_ID, Prediction
from app.ml.text import normalize
from app.services.agent.executor import QueryResult, SAFE_CONFIG
from app.services.data.ingest import TABLE_NAME

MIN_AVG_LENGTH = 15
MAX_GROUPS = 15
SAMPLES_PER_LABEL = 2
ORDER = ["positif", "netral", "negatif"]


class SentimentUnavailable(ValueError):
    pass


def _mentioned(column: str, question: str) -> bool:
    words = set(normalize(question).split())
    return all(part in words for part in column.split("_") if part)


def pick_columns(db_path: Path, question: str) -> tuple[str, str | None]:
    """Choose the free-text column to analyse and, if the question names one, a grouping column."""
    with duckdb.connect(str(db_path), read_only=True, config=SAFE_CONFIG) as con:
        varchar = [
            row[0]
            for row in con.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = ? AND data_type = 'VARCHAR'",
                [TABLE_NAME],
            ).fetchall()
        ]
        if not varchar:
            raise SentimentUnavailable("Dataset ini tidak memiliki kolom teks untuk dianalisis sentimennya.")
        stats = {}
        for col in varchar:
            avg_len, distinct = con.execute(
                f'SELECT avg(length("{col}")), approx_count_distinct("{col}") FROM {TABLE_NAME}'
            ).fetchone()
            stats[col] = (avg_len or 0, distinct or 0)

    text_candidates = [c for c in varchar if stats[c][0] >= MIN_AVG_LENGTH]
    if not text_candidates:
        raise SentimentUnavailable("Tidak ada kolom berisi kalimat (ulasan, komentar, keluhan) di dataset ini.")
    named = [c for c in text_candidates if _mentioned(c, question)]
    text_col = named[0] if named else max(text_candidates, key=lambda c: stats[c][0])

    groups = [
        c for c in varchar if c != text_col and 1 < stats[c][1] <= MAX_GROUPS and _mentioned(c, question)
    ]
    return text_col, (groups[0] if groups else None)


def analyze(db_path: Path, question: str, model) -> tuple[QueryResult, dict]:
    text_col, group_col = pick_columns(db_path, question)
    select = f'"{text_col}"' + (f', "{group_col}"' if group_col else "")
    with duckdb.connect(str(db_path), read_only=True, config=SAFE_CONFIG) as con:
        where = f'"{text_col}" IS NOT NULL AND length(trim("{text_col}")) > 0'
        total_rows = con.execute(f"SELECT count(*) FROM {TABLE_NAME} WHERE {where}").fetchone()[0]
        limit = model.max_texts
        # A repeatable random sample, so the same question gives the same answer.
        sample = f"USING SAMPLE reservoir({limit} ROWS) REPEATABLE (42)" if total_rows > limit else ""
        rows = con.execute(
            f"SELECT {select} FROM (SELECT * FROM {TABLE_NAME} WHERE {where}) AS teks {sample}"
        ).fetchall()
    if not rows:
        raise SentimentUnavailable(f"Kolom {text_col} kosong.")

    predictions: list[Prediction] = model.predict([r[0] for r in rows])
    labels = [SENTIMENT_LABELS_ID.get(p.label, p.label) for p in predictions]

    if group_col:
        counts: dict[str, Counter] = {}
        for row, label in zip(rows, labels):
            counts.setdefault(str(row[1]), Counter())[label] += 1
        ordered = sorted(counts.items(), key=lambda kv: kv[1]["negatif"] / sum(kv[1].values()), reverse=True)
        result = QueryResult(
            columns=[group_col, *ORDER, "persen_negatif"],
            types=["VARCHAR", "BIGINT", "BIGINT", "BIGINT", "DOUBLE"],
            rows=[
                [g, c["positif"], c["netral"], c["negatif"], round(100 * c["negatif"] / sum(c.values()), 1)]
                for g, c in ordered
            ],
        )
    else:
        overall = Counter(labels)
        result = QueryResult(
            columns=["sentimen", "jumlah", "persen"],
            types=["VARCHAR", "BIGINT", "DOUBLE"],
            rows=[[label, overall[label], round(100 * overall[label] / len(labels), 1)] for label in ORDER],
        )

    samples = []
    for label in ("negatif", "positif"):
        ranked = sorted(
            ((p.confidence, row[0]) for row, p, lab in zip(rows, predictions, labels) if lab == label), reverse=True
        )
        samples.extend({"label": label, "text": str(text)[:240]} for _, text in ranked[:SAMPLES_PER_LABEL])

    meta = {
        "text_column": text_col,
        "group_column": group_col,
        "analyzed": len(rows),
        "total": total_rows,
        "sampled": len(rows) < total_rows,
        "model": model.name,
        "samples": samples,
    }
    return result, meta
