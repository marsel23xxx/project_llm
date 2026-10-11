import pytest

from app.services.agent.guardrail import GuardrailError, validate_sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT produk, SUM(qty) AS total FROM data GROUP BY produk ORDER BY total DESC LIMIT 5;",
        "WITH bulanan AS (SELECT date_trunc('month', tanggal) AS bulan, SUM(total) AS omzet FROM data GROUP BY 1) "
        "SELECT * FROM bulanan ORDER BY bulan",
        "SELECT produk FROM data UNION SELECT 'lainnya'",
    ],
)
def test_allows_read_only_queries_on_data(sql):
    assert validate_sql(sql) == sql.strip().rstrip(";")


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("DROP TABLE data", "SELECT"),
        ("DELETE FROM data", "SELECT"),
        ("SELECT 1; DROP TABLE data", "satu query"),
        ("SELECT * FROM read_csv('/etc/passwd')", "read_csv"),
        ("SELECT * FROM read_text('/etc/passwd')", "read_text"),
        ("SELECT * FROM users", "tabel `data`"),
        ("SELECT * FROM other_db.main.data", "tabel `data`"),
        ("SELECT getenv('JWT_SECRET')", "getenv"),
        ("ATTACH 'x.db'", "SELECT"),
        ("COPY data TO 'out.csv'", "SELECT"),
        ("", "kosong"),
    ],
)
def test_rejects_unsafe_queries(sql, reason):
    with pytest.raises(GuardrailError, match=reason):
        validate_sql(sql)
