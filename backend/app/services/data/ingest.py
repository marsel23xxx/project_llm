"""Load user uploads (CSV/XLSX) into a per-dataset DuckDB file.

Indonesian spreadsheets often use ';' as the CSV delimiter, '.' as the thousands
separator, ',' as the decimal mark, and "Rp" prefixes, so these are normalised here
before the agent ever sees the data.
"""

import csv
import re
from pathlib import Path

import duckdb
import pandas as pd

SUPPORTED_SUFFIXES = {".csv", ".xlsx"}
TABLE_NAME = "data"

_ID_NUMBER = re.compile(r"^-?(?:rp\.?\s?)?\d{1,3}(?:\.\d{3})+(?:,\d+)?$|^-?(?:rp\.?\s?)?\d+,\d+$", re.IGNORECASE)
_DATE_HINT = re.compile(r"tanggal|tgl|date|waktu|periode")
# Share of non-null values that must match before a column is converted.
_MATCH_THRESHOLD = 0.9


class IngestError(ValueError):
    pass


def _is_text(series: pd.Series) -> bool:
    # pandas 3 reads text as the "str" dtype; older versions use object.
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def clean_column_names(columns: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: dict[str, int] = {}
    for i, raw in enumerate(columns):
        name = re.sub(r"[^0-9a-z]+", "_", str(raw).strip().lower()).strip("_")
        if not name or name.startswith("unnamed"):
            name = f"kolom_{i + 1}"
        if name[0].isdigit():
            name = f"k_{name}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        cleaned.append(name)
    return cleaned


def _parse_id_number(value: str) -> float:
    text = re.sub(r"(?i)^(-?)rp\.?\s?", r"\1", value.strip())
    return float(text.replace(".", "").replace(",", "."))


def normalize_indonesian_numbers(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if not _is_text(df[col]):
            continue
        values = df[col].dropna().astype(str).str.strip()
        if values.empty:
            continue
        if values.str.match(_ID_NUMBER).mean() >= _MATCH_THRESHOLD:
            df[col] = df[col].map(
                lambda v: _parse_id_number(str(v)) if pd.notna(v) and _ID_NUMBER.match(str(v).strip()) else None
            ).astype("Float64")
    return df


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if not _is_text(df[col]) or not _DATE_HINT.search(col):
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True, format="mixed")
        if parsed.notna().mean() >= _MATCH_THRESHOLD:
            df[col] = parsed
    return df


def _sniff_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        sample = f.read(64 * 1024)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def load_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise IngestError(f"Format {suffix} belum didukung. Gunakan CSV atau XLSX.")
    try:
        if suffix == ".csv":
            df = pd.read_csv(path, sep=_sniff_delimiter(path), encoding="utf-8-sig", dtype=str, keep_default_na=True)
        else:
            df = pd.read_excel(path, dtype=str)
    except Exception as exc:  # pandas raises many parser-specific types
        raise IngestError(f"File tidak bisa dibaca: {exc}") from exc

    if df.empty or len(df.columns) == 0:
        raise IngestError("File tidak berisi data.")

    df.columns = clean_column_names(list(df.columns))
    # Everything was read as text so Indonesian formats survive; now infer real types.
    df = normalize_indonesian_numbers(df)
    for col in df.columns:
        if _is_text(df[col]):
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() >= _MATCH_THRESHOLD * df[col].notna().sum() > 0:
                df[col] = converted
    return normalize_dates(df)


def write_duckdb(df: pd.DataFrame, db_path: Path) -> int:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as con:
        con.register("upload_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {TABLE_NAME} AS SELECT * FROM upload_df")
        con.unregister("upload_df")
        return con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
