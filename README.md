# Agentic Data Analyst

Asisten analisis data berbasis AI agent. User mengunggah CSV/Excel, lalu bertanya dalam Bahasa Indonesia; agent menulis SQL/Python, menjalankannya dengan aman, dan menjawab dengan tabel, grafik, dan insight.

> Status: **Fase 3a (Model ML)**: agent LangGraph (Text-to-SQL + Python sandbox + guardrail) dengan router intent scikit-learn dan analisis sentimen IndoBERT (PyTorch, ONNX int8). Forecasting LSTM (TensorFlow) dan benchmark optimasi menyusul.

## Cara kerja agent

```
pertanyaan ─► route ─┬─ sql ───────► generate_sql ────► execute_sql ────┬─► summarize ─► jawaban
   (model ML; Claude │                    ▲                              │
    jika ragu)      │                    └──── error: perbaiki (maks 3) ┤
                     ├─ python ────► generate_python ─► execute_python ─┤
                     ├─ sentiment ─► analyze_sentiment (model SmSA) ─────┘
                     └─ chat ──────► answer_chat
```

- **Guardrail SQL** (sqlglot): hanya satu `SELECT` pada tabel `data`, tanpa DDL/DML, tanpa fungsi baca file.
- **DuckDB read-only**: akses file eksternal dimatikan dan konfigurasinya dikunci, query dihentikan setelah 15 detik.
- **Python sandbox**: container terpisah tanpa internet, filesystem read-only, non-root, tanpa capability; setiap eksekusi berjalan di subprocess baru dengan batas CPU 30 detik, RAM 1,5 GB, dan file 10 MB.
- **Streaming**: setiap langkah agent (intent, kode, tabel, grafik, jawaban) dikirim lewat SSE dan disimpan agar riwayat chat tampil sama persis.

## Model ML

| Model | Framework | Tugas | Hasil |
| --- | --- | --- | --- |
| Intent router | Scikit-learn (TF-IDF kata + karakter, LinearSVC terkalibrasi) | Memilih jalur agent tanpa memanggil LLM | Akurasi 86.7% pada 98 pertanyaan tulisan tangan; di atas ambang 0.9 menangani 45% pertanyaan dengan presisi 100%, sisanya ke LLM; ~0,2 ms/pertanyaan |
| Sentimen (baseline) | Scikit-learn (TF-IDF + Logistic Regression) | Analisis sentimen kolom teks | IndoNLU SmSA: akurasi 88.6%, macro-F1 0.856; ~0,4 ms/teks |
| Sentimen (IndoBERT) | PyTorch (fine-tune) + ONNX Runtime int8 | Dipakai agent untuk analisis sentimen | IndoNLU SmSA: akurasi 94,2%, macro-F1 0,919 (baseline 88,7% / 0,856). Kuantisasi int8: 498 MB menjadi 125 MB tanpa penurunan akurasi, 164 ms menjadi 131 ms per teks di CPU 2 thread |

Catatan metodologi:
- Data intent dibuat dari template (`ml/intent/generate.py`). Pemilihan model dan ambang confidence memakai *template yang disisihkan* (GroupShuffleSplit), bukan test set, sehingga 98 pertanyaan manual (`ml/intent/heldout.csv`) tetap menjadi data yang belum pernah dilihat.
- Label test SmSA resmi disembunyikan, jadi split validasi resmi (1.260 teks) dipakai sebagai test set.
- Benchmark router pada 98 pertanyaan yang sama (`ml/benchmark_router.py`, laporan di `ml/reports/router_benchmark.json`):

  | Router | Akurasi | Latensi |
  | --- | --- | --- |
  | Model ML (scikit-learn) | 86,7% | ~5 ms |
  | LLM lokal Qwen2.5-Coder 3B | 57,1% | ~1.900 ms |
  | ML, ke Qwen jika ragu | 71,4% | - |

  Karena Qwen 3B lebih lemah dari classifier, router hanya bertanya ke LLM saat Claude API aktif; dengan LLM lokal saja, classifier selalu memutuskan.
- IndoBERT ~90 ms per teks di CPU, jadi untuk kolom di atas 300 teks agent menganalisis sampel acak 300 teks dan menyebutkannya di jawaban; baseline scikit-learn (0,7 ms per teks) menganalisis sampai 5.000 teks.
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
  app/services/llm/     provider lokal (llama.cpp) + Claude API, router hybrid
  app/services/agent/   graph LangGraph, guardrail, eksekutor DuckDB, klien sandbox, pemilih grafik, sentimen
  app/ml/               normalisasi teks Indonesia + pemuat model
  models/               model terlatih (intent, sentimen)
  tests/
sandbox/                service eksekusi Python terisolasi
ml/                     data, training, laporan, MLflow, notebook Colab IndoBERT
frontend/               Next.js: login, upload, chat streaming, tabel + grafik
scripts/download_model.sh   unduh model LLM lokal (~2 GB)
docker-compose.yml          postgres (pgvector), redis, llm, backend
```

## Menjalankan

```bash
cp .env.example .env          # lalu ganti JWT_SECRET dan POSTGRES_PASSWORD
docker compose up -d --build  # postgres, redis, backend
```

API berjalan di http://localhost:8000 (atau `BACKEND_PORT`) dengan dokumentasi interaktif di `/docs`.

Frontend (development):

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
| `hybrid` | Pertanyaan sederhana ke lokal, multi-langkah ke Claude API | sesuai pemakaian |
| `api` | Semua ke Claude API, lokal sebagai cadangan | sesuai pemakaian |

Gunakan `local` selama development. Isi `ANTHROPIC_API_KEY` dan pasang spend limit di Claude Console hanya saat siap live demo.

## Endpoint

| Method | Path | Fungsi |
| --- | --- | --- |
| POST | `/api/v1/auth/register` | Daftar akun (email + password) |
| POST | `/api/v1/auth/login` | Login, token disimpan di cookie httpOnly |
| POST | `/api/v1/auth/refresh` | Perpanjang sesi |
| POST | `/api/v1/auth/logout` | Keluar |
| GET | `/api/v1/auth/me` | Data user yang login |
| POST | `/api/v1/datasets` | Upload CSV/XLSX (maks 50 MB), return profil kolom |
| GET | `/api/v1/datasets` | Daftar dataset milik user |
| GET | `/api/v1/datasets/{id}/profile` | Statistik kolom + sampel baris |
| DELETE | `/api/v1/datasets/{id}` | Hapus dataset |
| POST | `/api/v1/chat/sessions` | Buat sesi chat untuk satu dataset |
| GET | `/api/v1/chat/sessions` | Daftar sesi |
| GET | `/api/v1/chat/sessions/{id}` | Riwayat pesan beserta langkah agent |
| POST | `/api/v1/chat/sessions/{id}/messages` | Kirim pertanyaan, jawaban di-stream (SSE); kuota harian per user |
| GET | `/api/v1/health` | Status database, redis, mode LLM |

## Tes

```bash
cd backend
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```
