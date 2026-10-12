# Agentic Data Analyst

Asisten analisis data berbasis AI agent. User membuka chat, melampirkan CSV/Excel, PDF, atau foto dokumen (bisa beberapa file per chat), lalu bertanya dalam Bahasa Indonesia; agent menulis SQL/Python, menjalankannya dengan aman, dan menjawab dengan tabel, grafik, dan insight.

> Status: agent LangGraph dalam sesi chat dengan banyak lampiran: data tabel (SQL, Python sandbox, sentimen IndoBERT, forecasting LSTM), dokumen PDF, dan gambar/PDF hasil scan lewat OCR. Laporan PDF, LLM lokal atau API (DeepSeek/Claude), live di VPS. Berikutnya: CI/CD dan monitoring.

## Cara kerja agent

```
pertanyaan ─► route ─┬─ sql ───────► generate_sql ────► execute_sql ────┬─► summarize ─► jawaban
   (model ML; Claude │                    ▲                              │
    jika ragu)      │                    └──── error: perbaiki (maks 3) ┤
                     ├─ python ────► generate_python ─► execute_python ─┤
                     ├─ sentiment ─► analyze_sentiment (IndoBERT) ───────┘
                     ├─ forecast ──► run_forecast (LSTM vs baseline, backtest) ─► jawaban
                     └─ chat ──────► answer_chat
```

- **Guardrail SQL** (sqlglot): hanya satu `SELECT` pada tabel `data`, tanpa DDL/DML, tanpa fungsi baca file.
- **DuckDB read-only**: akses file eksternal dimatikan dan konfigurasinya dikunci, query dihentikan setelah 15 detik.
- **Python sandbox**: container terpisah tanpa internet, filesystem read-only, non-root, tanpa capability; setiap eksekusi berjalan di subprocess baru dengan batas CPU 30 detik, RAM 1,5 GB, dan file 10 MB.
- **Forecast service**: container TensorFlow terpisah tanpa internet dan read-only; hanya menerima deret angka hasil agregasi, bukan data mentah.
- **Laporan PDF**: setiap jawaban bisa diunduh sebagai PDF (pertanyaan, jawaban, grafik, tabel, cara perhitungan) yang dibuat server dari langkah agent yang tersimpan, bukan oleh LLM. Permintaan seperti "buatkan dalam PDF" dijawab normal lalu diarahkan ke tombol ini.
- **Ringkasan dataset**: pertanyaan "ringkas isi data ini" dijawab dari profil kolom tanpa LLM.
- **Streaming**: setiap langkah agent (intent, kode, tabel, grafik, jawaban) dikirim lewat SSE dan disimpan agar riwayat chat tampil sama persis.

## Sesi chat dan lampiran

Satu chat bisa dipakai tanpa file, dan bisa dilampiri sampai 10 file kapan saja (tombol lampiran, atau tempel gambar dari clipboard).

- **Dokumen dan gambar** di satu chat dibaca sebagai satu kumpulan; jawaban menyebut nama file dan halamannya.
- **OCR lebih dulu, vision sebagai cadangan.** Gambar PNG/JPG/WEBP dan halaman PDF tanpa teks dibaca dengan Tesseract (model Indonesia + Inggris) di server: gratis dan gambarnya tidak keluar. Tesseract memberi keyakinan per kata; jika rata-ratanya di bawah 85 dan API hosted aktif, gambar itu dikirim ke model vision (`deepseek-flash`, maksimal 1.024 token per gambar). Jika vision gagal, hasil OCR tetap dipakai. `VISION_FALLBACK=false` mematikan pengiriman gambar. Halaman PDF yang punya teks asli tidak di-OCR. Batas 30 halaman scan per PDF.
- Batas 85 berasal dari pengukuran (`ml/benchmark_ocr.py`, 9 gambar dari teks yang diketahui): OCR membaca semua angka dengan benar pada 5 gambar (bersih, miring 4 derajat, buram ringan, resolusi rendah, gelap), semuanya dengan keyakinan 90 ke atas; 4 gambar yang gagal (miring 8 dan 15 derajat, buram berat, berbintik) semuanya di bawah 85. Kemampuan vision pada 4 gambar itu diukur dengan script yang sama saat API key tersedia. Gambarnya sintetis; foto HP sungguhan lebih sulit.
- **Tabel**: jika satu chat berisi beberapa tabel, yang dianalisis adalah yang terakhir dilampirkan. Menggabungkan beberapa tabel belum didukung.
- **Chat berisi tabel dan dokumen**: kata di pertanyaan menentukan tujuannya ("dokumen", "halaman", "kontrak", "gambar" mengarah ke dokumen; selain itu ke tabel).
- Menghapus satu file tidak menghapus percakapannya; menghapus chat menghapus semua file dan pesannya.

## Dokumen PDF (RAG)

PDF dipecah per halaman menjadi potongan 110 kata. Untuk setiap pertanyaan, 3 potongan paling relevan dicari dengan TF-IDF (kata + n-gram karakter, tahan typo dan slang), lalu LLM menjawab hanya dari kutipan itu. Halaman sumber ditampilkan oleh aplikasi dari hasil pencarian, bukan ditulis oleh LLM.

Evaluasi pada `samples/sop_layanan_linen.pdf` dengan 14 pertanyaan berkunci jawaban (`ml/benchmark_documents.py`, model qwen2.5-coder-3b-instruct di CPU):

| Ukuran | Hasil |
| --- | --- |
| Halaman yang benar masuk 3 teratas | 12/12 |
| Halaman yang benar di peringkat 1 | 12/12 |
| Jawaban benar (jawaban ada di dokumen) | 11/12 |
| Menolak menjawab saat jawaban memang tidak ada | 2/2 |
| Waktu per jawaban | median 20.1 detik, maksimum 40.5 detik |

Keputusan desain dan batasannya:
- Dengan model hosted (DeepSeek/Claude), dokumen sampai 20.000 kata dikirim utuh di setiap pertanyaan, karena model hosted membaca prompt panjang dalam hitungan detik; ini yang memungkinkan permintaan seperti "apa saranmu untuk dokumen ini". Dokumen diletakkan di awal prompt agar pertanyaan berikutnya terkena prompt cache penyedia. Di atas batas itu, atau dengan LLM lokal, jalur pencarian kutipan di bawah yang dipakai.
- Di sesi dokumen, jalur dipilih dengan aturan (ringkasan, telaah, tanya), bukan classifier intent: classifier dilatih pada pertanyaan tabel dan mengira "apa saranmu terhadap dokumen ini" sebagai obrolan.
- LLM di CPU membaca sekitar 15 sampai 20 token prompt per detik, jadi ukuran prompt adalah latensi. Versi pertama (4 potongan 170 kata) butuh 42 sampai 171 detik dan sering menjawab "tidak ditemukan"; potongan lebih kecil, lebih sedikit, dan urutan terkuat di dekat pertanyaan menurunkannya ke sekitar 20 detik dengan jawaban yang lebih tepat.
- Ringkasan dokumen panjang dibuat dari 12 cuplikan yang tersebar merata di seluruh halaman, dan jawaban menyebutkan hal itu.
- Satu kegagalan yang tersisa pada evaluasi adalah model 3B menjawab "tidak ditemukan" walau kutipannya benar; prompt tidak disetel khusus untuk pertanyaan itu.
- Evaluasi ini memakai satu dokumen sintetis yang pendek. Dokumen nyata yang panjang, bertabel rumit, atau hasil scan (ditolak, belum ada OCR) belum terukur.
- Pencarian masih leksikal. Embedding + pgvector adalah langkah berikutnya untuk pertanyaan yang memakai kata berbeda dari dokumen.

## Model ML

| Model | Framework | Tugas | Hasil |
| --- | --- | --- | --- |
| Intent router | Scikit-learn (TF-IDF kata + karakter, LinearSVC terkalibrasi) | Memilih 1 dari 5 jalur agent tanpa memanggil LLM | Akurasi 87,3% pada 110 pertanyaan tulisan tangan; di atas ambang 0.9 menangani 37% pertanyaan dengan presisi 100%; ~5 ms/pertanyaan |
| Sentimen (baseline) | Scikit-learn (TF-IDF + Logistic Regression) | Analisis sentimen kolom teks | IndoNLU SmSA: akurasi 88.6%, macro-F1 0.856; ~0,4 ms/teks |
| Sentimen (IndoBERT) | PyTorch (fine-tune) + ONNX Runtime int8 | Dipakai agent untuk analisis sentimen | IndoNLU SmSA: akurasi 94,2%, macro-F1 0,919 (baseline 88,7% / 0,856). Kuantisasi int8: 498 MB menjadi 125 MB tanpa penurunan akurasi, 164 ms menjadi 131 ms per teks di CPU 2 thread |
| Forecasting | TensorFlow/Keras (LSTM) vs baseline naive | Prediksi deret waktu dari data user | Pemenang dipilih lewat backtest di setiap permintaan; lihat benchmark di bawah |

Catatan metodologi:
- Data intent dibuat dari template (`ml/intent/generate.py`). Pemilihan model dan ambang confidence memakai *template yang disisihkan* (GroupShuffleSplit), bukan test set, sehingga 110 pertanyaan manual (`ml/intent/heldout.csv`) tetap menjadi data yang belum pernah dilihat.
- Label test SmSA resmi disembunyikan, jadi split validasi resmi (1.260 teks) dipakai sebagai test set.
- Benchmark router pada 110 pertanyaan yang sama (`ml/benchmark_router.py`, laporan di `ml/reports/router_benchmark.json`):

  | Router | Akurasi | Latensi |
  | --- | --- | --- |
  | Model ML (scikit-learn) | 87,3% | ~5 ms |
  | LLM lokal Qwen2.5-Coder 3B | 60,9% | ~1.600 ms |
  | ML, ke Qwen jika ragu | 72,7% | - |

  Karena Qwen 3B lebih lemah dari classifier, router hanya bertanya ke LLM saat Claude API aktif; dengan LLM lokal saja, classifier selalu memutuskan.
- IndoBERT ~90 ms per teks di CPU, jadi untuk kolom di atas 300 teks agent menganalisis sampel acak 300 teks dan menyebutkannya di jawaban; baseline scikit-learn (0,7 ms per teks) menganalisis sampai 5.000 teks.
- Benchmark forecasting pada dataset publik (`ml/benchmark_forecast.py`, rolling origin 3 titik potong, MAE; makin kecil makin baik):

  | Dataset | Naive | Moving average | Seasonal naive | LSTM | Terbaik |
  | --- | --- | --- | --- | --- | --- |
  | Penumpang pesawat bulanan (144 titik) | 73,22 | 57,34 | 35,92 | 18,65 | lstm |
  | Penjualan mobil bulanan (108 titik) | 3.615,42 | 3.243,18 | 1.461,72 | 2.854,97 | seasonal_naive |
  | Kelahiran harian (365 titik) | 5,81 | 6,94 | 7,86 | 6,76 | naive |
  | Suhu minimum harian (600 titik terakhir) | 2,47 | 2 | 2,52 | 2,34 | moving_average |

  LSTM unggul jelas hanya pada deret dengan tren kuat dan pola musiman (galat 48% lebih kecil dari baseline terbaik), dan kalah di tiga lainnya. Karena itu service tidak mengasumsikan LSTM terbaik: setiap permintaan menyisihkan periode terakhir, menguji semua kandidat, lalu memakai yang galatnya terkecil. LSTM tidak di-tuning terhadap dataset benchmark ini.
- Jawaban prediksi disusun langsung dari angka hasil model, tanpa LLM, sehingga tidak ada angka yang dikarang.
- Semua eksperimen tercatat di MLflow (`ml/mlflow.db`): `cd ml && .venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db`.

Melatih ulang:

```bash
cd ml
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python intent/generate.py && .venv/bin/python train_intent.py
.venv/bin/python train_sentiment_baseline.py   # butuh data/smsa (lihat notebook Colab untuk URL)
```

IndoBERT (model 125 MB tidak disimpan di git): buka `ml/colab/indobert_sentiment.ipynb` di Google Colab (GPU T4), pilih Runtime > Run all, lalu ekstrak `sentiment_onnx.zip` ke `backend/models/sentiment_onnx/`. Backend otomatis memakai IndoBERT saat file tersebut ada.

## Struktur

```
backend/                FastAPI
  app/api/v1/           endpoint: auth, datasets, health
  app/core/             config, keamanan (Argon2 + JWT), rate limit
  app/db/               model SQLAlchemy (users, datasets)
  app/services/data/    ingest CSV/XLSX ke DuckDB, profiling, cleanup 24 jam
  app/services/llm/     provider lokal (llama.cpp), OpenAI-compatible (DeepSeek), Claude; router hybrid
  app/services/agent/   graph LangGraph, guardrail, eksekutor DuckDB, klien sandbox, pemilih grafik, sentimen
  app/services/documents.py   ekstraksi PDF/gambar, pemecahan teks, pencarian kutipan
  app/services/ocr.py         OCR Tesseract untuk gambar dan halaman scan
  app/services/vision.py      model vision untuk gambar yang tidak terbaca jelas oleh OCR
  app/services/uploads.py     penyimpanan file yang dilampirkan
  app/db/migrate.py           migrasi skema ringan saat startup
  app/services/report.py      pembuat laporan PDF
  app/ml/               normalisasi teks Indonesia + pemuat model
  models/               model terlatih (intent, sentimen)
  tests/
sandbox/                service eksekusi Python terisolasi
forecast/               service forecasting (TensorFlow LSTM vs baseline)
samples/                data contoh sintetis untuk demo
deploy/                 panduan deploy VPS dan konfigurasi Nginx
ml/                     data, training, laporan, MLflow, notebook Colab IndoBERT
frontend/               Next.js: login, upload, chat streaming, tabel + grafik
scripts/download_model.sh   unduh model LLM lokal (~2 GB)
docker-compose.yml          postgres (pgvector), redis, llm, backend, sandbox, forecast, frontend
```

## Menjalankan

```bash
cp .env.example .env          # lalu ganti JWT_SECRET dan POSTGRES_PASSWORD
docker compose up -d --build  # postgres, redis, backend, sandbox, forecast, frontend
```

Aplikasi berjalan di http://localhost:3000 (`FRONTEND_PORT`), API di http://localhost:8000 (`BACKEND_PORT`) dengan dokumentasi interaktif di `/docs`.

Deploy ke VPS dengan Nginx dan HTTPS: lihat [deploy/DEPLOY.md](deploy/DEPLOY.md).

Frontend dengan hot reload (development):

```bash
cd frontend
npm install
BACKEND_URL=http://localhost:8000 npm run dev
```

LLM lokal (gratis, butuh ~3 GB RAM):

```bash
./scripts/download_model.sh
docker compose --profile llm up -d
```

## Mode LLM dan biaya

| `LLM_MODE` | Perilaku | Biaya API |
| --- | --- | --- |
| `local` | Semua jawaban dari Qwen2.5-Coder 3B di VPS | US$0 |
| `hybrid` | Tugas ringan (merangkum hasil) ke lokal; SQL, kode, dan dokumen ke API | sesuai pemakaian |
| `api` | Semua ke API, lokal hanya sebagai cadangan jika API gagal atau saldo habis | sesuai pemakaian |

"API" adalah model hosted yang dikonfigurasi: DeepSeek (atau penyedia lain berformat OpenAI) lewat `REMOTE_LLM_API_KEY`, atau Claude lewat `ANTHROPIC_API_KEY` (Claude didahulukan jika keduanya diisi). Yang dikirim ke penyedia hanya isi prompt agent: nama kolom, 3 baris contoh, atau kutipan dokumen yang relevan, bukan file yang diunggah.

Gunakan `local` selama development. API key baru diisi saat siap live demo; kedua penyedia memakai saldo prabayar.

## Endpoint

| Method | Path | Fungsi |
| --- | --- | --- |
| POST | `/api/v1/auth/register` | Daftar akun (email + password) |
| POST | `/api/v1/auth/login` | Login, token disimpan di cookie httpOnly |
| POST | `/api/v1/auth/refresh` | Perpanjang sesi |
| POST | `/api/v1/auth/logout` | Keluar |
| GET | `/api/v1/auth/me` | Data user yang login |
| GET | `/api/v1/datasets/{id}/profile` | Statistik kolom + sampel baris |
| DELETE | `/api/v1/datasets/{id}` | Hapus satu file dari chat |
| POST | `/api/v1/chat/sessions` | Buat chat kosong |
| POST | `/api/v1/chat/sessions/{id}/files` | Lampirkan CSV/XLSX/PDF/gambar ke chat (maks 50 MB) |
| DELETE | `/api/v1/chat/sessions/{id}` | Hapus chat beserta file dan pesannya |
| GET | `/api/v1/chat/sessions` | Daftar sesi |
| GET | `/api/v1/chat/sessions/{id}` | Riwayat pesan beserta langkah agent |
| POST | `/api/v1/chat/sessions/{id}/messages` | Kirim pertanyaan, jawaban di-stream (SSE); kuota harian per user |
| GET | `/api/v1/chat/messages/{id}/pdf` | Unduh laporan PDF untuk satu jawaban |
| GET | `/api/v1/health` | Status database, redis, mode LLM |

## Tes

```bash
cd backend
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```
