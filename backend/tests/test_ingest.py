from pathlib import Path

import pandas as pd

from app.services.data.ingest import clean_column_names, load_dataframe, normalize_indonesian_numbers, write_duckdb
from app.services.data.profiling import profile_table


def test_clean_column_names_handles_spaces_duplicates_and_blanks():
    assert clean_column_names(["Nama Produk", "Harga (Rp)", "nama produk", "", "2024"]) == [
        "nama_produk",
        "harga_rp",
        "nama_produk_2",
        "kolom_4",
        "k_2024",
    ]


def test_indonesian_number_formats_become_floats():
    df = pd.DataFrame({"harga": ["Rp 1.500.000", "2.750,50", "300,25", None], "kota": ["Bandung", "Medan", "Solo", "Bogor"]})
    out = normalize_indonesian_numbers(df)
    assert out["harga"].tolist()[:3] == [1500000.0, 2750.5, 300.25]
    assert pd.isna(out["harga"].iloc[3])
    assert out["kota"].tolist() == ["Bandung", "Medan", "Solo", "Bogor"]


def test_semicolon_csv_is_loaded_typed_and_profiled(tmp_path: Path):
    csv_path = tmp_path / "penjualan.csv"
    csv_path.write_text(
        "Tanggal;Produk;Qty;Total\n"
        "01/09/2026;Kopi Susu;3;Rp 54.000\n"
        "02/09/2026;Teh Manis;5;Rp 40.000\n"
        "15/09/2026;Kopi Susu;2;Rp 36.000\n",
        encoding="utf-8",
    )
    df = load_dataframe(csv_path)
    assert list(df.columns) == ["tanggal", "produk", "qty", "total"]
    assert pd.api.types.is_datetime64_any_dtype(df["tanggal"])
    assert df["tanggal"].iloc[2].day == 15
    assert df["qty"].sum() == 10
    assert df["total"].sum() == 130000.0

    db_path = tmp_path / "ds.duckdb"
    assert write_duckdb(df, db_path) == 3
    profile = profile_table(db_path)
    assert profile["row_count"] == 3
    assert {c["name"] for c in profile["columns"]} == {"tanggal", "produk", "qty", "total"}
    assert len(profile["sample"]) == 3


def test_title_rows_above_the_table_are_skipped(tmp_path: Path):
    xlsx = tmp_path / "linen.xlsx"
    rows = [
        ["REKAP PENERIMAAN DAN PENGIRIMAN LINEN", None, None, None],
        ["Periode: September 2026", None, None, None],
        [None, None, None, None],
        ["No", "Nama Linen", "Jumlah Terima", "Tanggal"],
        [1, "Sprei", "1.250", "01/09/2026"],
        [2, "Selimut", "830", "02/09/2026"],
        [3, "Sarung Bantal", "2.100", "03/09/2026"],
    ]
    pd.DataFrame(rows).to_excel(xlsx, header=False, index=False)
    df = load_dataframe(xlsx)
    assert list(df.columns) == ["no", "nama_linen", "jumlah_terima", "tanggal"]
    assert len(df) == 3 and df["jumlah_terima"].sum() == 4180
    assert pd.api.types.is_datetime64_any_dtype(df["tanggal"])


def test_csv_with_title_line_and_blank_line(tmp_path: Path):
    path = tmp_path / "laporan.csv"
    path.write_text("Laporan Penjualan 2026\n\nProduk,Qty,Total\nKopi,3,54000\nTeh,5,40000\n", encoding="utf-8")
    df = load_dataframe(path)
    assert list(df.columns) == ["produk", "qty", "total"]
    assert df["qty"].tolist() == [3, 5]


def test_header_cell_left_blank_gets_a_placeholder_name(tmp_path: Path):
    path = tmp_path / "kosong.csv"
    path.write_text("Nama,,Nilai\nA,x,1\nB,y,2\n", encoding="utf-8")
    assert list(load_dataframe(path).columns) == ["nama", "kolom_2", "nilai"]


def test_thousands_dots_mixed_with_plain_integers():
    df = normalize_indonesian_numbers(pd.DataFrame({"jumlah": ["1.250", "830", "2.100", "45"]}))
    assert df["jumlah"].tolist() == [1250.0, 830.0, 2100.0, 45.0]


def test_real_decimals_are_not_read_as_thousands():
    df = pd.DataFrame({"rasio": ["1.250", "0.875", "2.100"]})
    out = normalize_indonesian_numbers(df.copy())
    assert out["rasio"].tolist() == ["1.250", "0.875", "2.100"]  # left for the plain numeric parser
