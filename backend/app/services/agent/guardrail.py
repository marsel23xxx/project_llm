"""First line of defence for agent-written SQL.

The DuckDB connection itself is the second line (read-only, external access disabled,
configuration locked), so a query that slips past this parser still cannot touch files
or modify data.
"""

import sqlglot
from sqlglot import exp

from app.services.data.ingest import TABLE_NAME

_FORBIDDEN_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
    exp.Copy,
    exp.Attach,
    exp.Detach,
    exp.Pragma,
    exp.Set,
    exp.Use,
    exp.Transaction,
)
_FORBIDDEN_FUNCTIONS = {
    "read_csv",
    "read_csv_auto",
    "read_parquet",
    "parquet_scan",
    "read_json",
    "read_json_auto",
    "read_ndjson",
    "read_text",
    "read_blob",
    "glob",
    "sniff_csv",
    "query",
    "query_table",
    "getenv",
}


class GuardrailError(ValueError):
    pass


def _function_name(node: exp.Expression) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.this).lower()
    return node.sql_name().lower()


def validate_sql(sql: str) -> str:
    """Return the cleaned query, or raise GuardrailError explaining why it was rejected."""
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise GuardrailError("Query kosong")
    try:
        statements = [s for s in sqlglot.parse(cleaned, read="duckdb") if s is not None]
    except sqlglot.errors.ParseError as exc:
        raise GuardrailError(f"SQL tidak valid: {exc}") from exc

    if len(statements) != 1:
        raise GuardrailError("Hanya boleh satu query")
    statement = statements[0]
    if not isinstance(statement, exp.Query):
        raise GuardrailError("Hanya query SELECT yang diizinkan")

    for node in statement.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise GuardrailError(f"Perintah {node.key.upper()} tidak diizinkan")
        if isinstance(node, exp.Func) and _function_name(node) in _FORBIDDEN_FUNCTIONS:
            raise GuardrailError(f"Fungsi {_function_name(node)} tidak diizinkan")

    cte_names = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE)}
    for table in statement.find_all(exp.Table):
        name = table.name.lower()
        if table.args.get("db") or table.args.get("catalog") or name not in cte_names | {TABLE_NAME}:
            raise GuardrailError(f"Hanya tabel `{TABLE_NAME}` yang boleh dipakai (ditemukan `{table.sql()}`)")

    return cleaned
