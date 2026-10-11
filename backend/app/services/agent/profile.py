"""Answer "what is in this dataset?" from the column profile, without an LLM."""

import re
from pathlib import Path

from app.ml.text import normalize
from app.services.agent.executor import QueryResult, to_jsonable
from app.services.data.profiling import profile_table

_ASKS_FOR_OVERVIEW = re.compile(
    r"\b(ringkas|ringkasan|rangkum|rangkuman|gambaran|deskripsi|deskripsikan|profil)\b.*"
    r"\b(data|dataset|file|tabel|list|daftar|isi)\b"
    r"|\bkolom apa (saja|aja)\b|\bisi (data|dataset|file|tabel)\b"
)
_TYPE_LABEL = (("TIMESTAMP", "tanggal"), ("DATE", "tanggal"), ("INT", "angka bulat"), ("DOUBLE", "angka desimal"),
               ("FLOAT", "angka desimal"), ("DECIMAL", "angka desimal"), ("BOOL", "ya/tidak"), ("VARCHAR", "teks"))  # fmt: skip
LIST_LIMIT = 6


def is_overview_question(question: str) -> bool:
    text = normalize(question)
    # "rangkum penjualan per kategori" is an aggregation, not a request to describe the file.
    return bool(_ASKS_FOR_OVERVIEW.search(text)) and " per " not in f" {text} "


def _label(duck_type: str) -> str:
    return next((label for key, label in _TYPE_LABEL if key in duck_type.upper()), duck_type.lower())


def _names(columns: list[dict]) -> str:
    names = [c["name"] for c in columns]
    extra = f" dan {len(names) - LIST_LIMIT} lainnya" if len(names) > LIST_LIMIT else ""
    return ", ".join(names[:LIST_LIMIT]) + extra


def describe(db_path: Path) -> tuple[QueryResult, str]:
    profile = profile_table(db_path)
    columns = profile["columns"]
    table = QueryResult(
        columns=["kolom", "jenis", "persen_kosong", "nilai_unik", "terkecil", "terbesar"],
        types=["VARCHAR", "VARCHAR", "DOUBLE", "BIGINT", "VARCHAR", "VARCHAR"],
        rows=[
            [c["name"], _label(c["type"]), c["null_pct"], to_jsonable(c["approx_unique"]),
             None if c["min"] is None else str(c["min"])[:40], None if c["max"] is None else str(c["max"])[:40]]
            for c in columns
        ],  # fmt: skip
    )

    by_kind: dict[str, list[dict]] = {}
    for c in columns:
        by_kind.setdefault(_label(c["type"]), []).append(c)
    rows = f"{profile['row_count']:,}".replace(",", ".")
    parts = [f"Dataset ini berisi {rows} baris dan {len(columns)} kolom."]
    numeric = by_kind.get("angka bulat", []) + by_kind.get("angka desimal", [])
    if numeric:
        parts.append(f"Kolom angka: {_names(numeric)}.")
    for c in by_kind.get("tanggal", [])[:1]:
        parts.append(f"Kolom tanggal {c['name']} mencakup {str(c['min'])[:10]} sampai {str(c['max'])[:10]}.")
    if by_kind.get("teks"):
        parts.append(f"Kolom teks: {_names(by_kind['teks'])}.")
    gaps = sorted((c for c in columns if c["null_pct"] > 0), key=lambda c: -c["null_pct"])
    if gaps:
        parts.append("Data kosong terbanyak ada di " + ", ".join(f"{c['name']} ({c['null_pct']:g}%)" for c in gaps[:3]) + ".")
    else:
        parts.append("Tidak ada sel kosong.")
    return table, " ".join(parts)
