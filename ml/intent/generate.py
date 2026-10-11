"""Generate a synthetic training set for the intent classifier.

Each row keeps the id of the template it came from, so validation can hold out whole
templates (GroupShuffleSplit) instead of near-duplicates of training rows.

Usage: python intent/generate.py  ->  data/intent/train.csv
"""

import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "intent" / "train.csv"
SAMPLES_PER_TEMPLATE = 14
SEED = 42

SLOTS = {
    "angka": ["penjualan", "omzet", "qty", "harga", "laba", "pendapatan", "jumlah pengaduan", "nilai", "biaya",
              "stok", "jumlah transaksi", "total", "skor", "gaji", "durasi layanan", "anggaran", "realisasi"],
    "kategori": ["produk", "kota", "provinsi", "cabang", "kategori", "layanan", "dinas", "pelanggan", "toko",
                 "wilayah", "kecamatan", "sales", "merek", "jenis kelamin", "segmen", "channel"],
    "waktu": ["bulan", "minggu", "tahun", "hari", "kuartal", "tanggal", "periode"],
    "teks": ["ulasan", "komentar", "review", "keluhan", "isi pengaduan", "feedback", "tweet", "testimoni",
             "masukan pelanggan", "kolom deskripsi", "saran", "tanggapan warga"],
    "nilai": ["Jakarta", "Bandung", "Surabaya", "kopi susu", "elektronik", "Jawa Barat", "online", "premium"],
    "n": ["3", "5", "10", "lima", "sepuluh", "tiga"],
    "periode": ["bulan lalu", "tahun ini", "2025", "kuartal pertama", "minggu ini", "september", "semester 1"],
}

TEMPLATES = {
    "sql": [
        "berapa total {angka} per {kategori}",
        "{kategori} apa yang {angka}nya paling tinggi",
        "tampilkan {n} {kategori} dengan {angka} terbesar",
        "urutkan {kategori} berdasarkan {angka}",
        "bagaimana tren {angka} per {waktu}",
        "grafik {angka} per {waktu} dong",
        "buatkan grafik {angka} per {kategori}",
        "berapa rata-rata {angka} di {nilai}",
        "berapa jumlah data untuk {kategori} {nilai}",
        "bandingkan {angka} antar {kategori}",
        "{angka} {periode} berapa",
        "total {angka} {periode} di {nilai}",
        "ada berapa {kategori} yang berbeda",
        "{kategori} mana yang paling sedikit {angka}nya",
        "tampilkan data {kategori} {nilai} saja",
        "berapa persen kontribusi tiap {kategori} terhadap total {angka}",
        "kenaikan {angka} dari {waktu} ke {waktu}",
        "{angka} tertinggi terjadi di {waktu} apa",
        "hitung {angka} maksimum dan minimum per {kategori}",
        "berapa banyak baris di data ini",
        "rekap {angka} per {kategori} per {waktu}",
        "siapa {kategori} terbaik {periode}",
        "produk terlaris {periode} apa",
        "filter {kategori} yang {angka}nya di atas rata-rata",
        "{n} {kategori} terbawah berdasarkan {angka}",
        "jumlah {angka} per {waktu} untuk {nilai}",
        "berapa {angka} kumulatif sampai {periode}",
        "ringkasan {angka} per {kategori}",
    ],
    "python": [
        "apakah ada korelasi antara {angka} dan {angka}",
        "hitung korelasi {angka} dengan {angka}",
        "seberapa kuat hubungan {angka} dan {angka}",
        "deteksi outlier pada {angka}",
        "apakah ada data anomali di {angka}",
        "lakukan clustering {kategori} berdasarkan {angka} dan {angka}",
        "segmentasikan {kategori} menjadi beberapa kelompok",
        "buat model regresi untuk memprediksi {angka} dari {angka}",
        "bagaimana distribusi {angka}",
        "apakah {angka} berdistribusi normal",
        "uji apakah {angka} {nilai} berbeda signifikan dengan yang lain",
        "lakukan uji t untuk {angka} antar {kategori}",
        "hitung skewness dan kurtosis {angka}",
        "analisis statistik deskriptif lengkap untuk {angka}",
        "faktor apa yang paling berpengaruh terhadap {angka}",
        "cari pola tersembunyi dari {angka} dan {kategori}",
        "kelompokkan {kategori} dengan k-means",
        "hitung matriks korelasi semua kolom angka",
        "apakah {angka} dan {angka} saling mempengaruhi secara statistik",
        "uji chi square antara {kategori} dan {kategori}",
        "nilai mana yang menyimpang jauh dari pola {angka}",
        "lakukan analisis regresi linear {angka}",
    ],
    "sentiment": [
        "analisis sentimen {teks}",
        "bagaimana sentimen {teks} pelanggan",
        "berapa persen {teks} yang negatif",
        "apakah {teks} kebanyakan positif atau negatif",
        "klasifikasikan {teks} jadi positif negatif netral",
        "apa kata pelanggan di {teks}",
        "bagaimana opini orang di kolom {teks}",
        "cek mood dari {teks}",
        "sentimen {teks} per {kategori}",
        "{kategori} mana yang paling banyak {teks} negatif",
        "pelanggan puas atau tidak berdasarkan {teks}",
        "seberapa banyak {teks} yang bernada positif",
        "tunjukkan perbandingan sentimen {teks} antar {kategori}",
        "nilai kepuasan dari {teks} gimana",
        "analisis emosi dari {teks}",
        "apakah {teks} {periode} lebih negatif",
    ],
    "chat": [
        "halo",
        "hai selamat pagi",
        "kamu bisa apa saja",
        "apa yang bisa kamu bantu",
        "terima kasih ya",
        "makasih banyak bantuannya",
        "siapa kamu",
        "bagaimana cara pakai aplikasi ini",
        "oke mantap",
        "cuaca hari ini bagaimana",
        "ceritakan lelucon",
        "apa itu data science",
        "kamu pakai model apa",
        "jelaskan cara kerja kamu",
        "data saya aman tidak",
        "bisa bahasa inggris tidak",
        "selamat malam",
        "tolong bantu saya",
        "sip paham",
        "kenapa jawabanmu lama",
    ],
}

PREFIXES = ["", "", "", "tolong ", "coba ", "bisa ", "saya mau tahu ", "min ", "kak ", "bang "]
SUFFIXES = ["", "", "", "?", " dong", " ya", " sih", " please", " ?", " gak"]
SLANGIFY = {"berapa": "brp", "yang": "yg", "tidak": "gak", "dengan": "dgn", "untuk": "utk", "bagaimana": "gimana",
            "rata-rata": "rata2", "jumlah": "jml", "bulan": "bln", "sudah": "udah"}


def _typo(text: str, rng: random.Random) -> str:
    words = text.split()
    i = rng.randrange(len(words))
    w = words[i]
    if len(w) > 4:
        j = rng.randrange(1, len(w) - 1)
        words[i] = w[:j] + w[j + 1 :]
    return " ".join(words)


def render(template: str, rng: random.Random) -> str:
    text = template
    while "{" in text:
        start = text.index("{")
        slot = text[start + 1 : text.index("}", start)]
        text = text.replace("{" + slot + "}", rng.choice(SLOTS[slot]), 1)
    if rng.random() < 0.3:
        text = " ".join(SLANGIFY.get(w, w) for w in text.split())
    if rng.random() < 0.15:
        text = _typo(text, rng)
    text = rng.choice(PREFIXES) + text + rng.choice(SUFFIXES)
    if rng.random() < 0.2:
        text = text.capitalize()
    return text.strip()


def main() -> None:
    rng = random.Random(SEED)
    rows = []
    for label, templates in TEMPLATES.items():
        for t_id, template in enumerate(templates):
            seen = set()
            for _ in range(SAMPLES_PER_TEMPLATE * 3):
                text = render(template, rng)
                if text not in seen:
                    seen.add(text)
                    rows.append({"text": text, "label": label, "group": f"{label}-{t_id}"})
                if len(seen) == SAMPLES_PER_TEMPLATE:
                    break
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "group"])
        writer.writeheader()
        writer.writerows(rows)
    counts = {label: sum(r["label"] == label for r in rows) for label in TEMPLATES}
    print(f"{len(rows)} contoh -> {OUT}  {counts}")


if __name__ == "__main__":
    main()
