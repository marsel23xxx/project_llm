# Deploy ke VPS

Panduan untuk VPS Ubuntu 22.04/24.04 (8 core, 12 GB RAM) dengan domain sendiri.
Ganti `DOMAIN_KAMU` dengan domainmu (misalnya `analyst.contoh.com`) dan `IP_VPS` dengan alamat IP VPS.

Susunan di server:

```
Internet ──► Nginx (host, port 80/443, SSL dari Certbot)
               ├─ /api/  ──► backend   (127.0.0.1:8000, Docker)
               └─ /      ──► frontend  (127.0.0.1:3000, Docker)
backend ──► postgres, redis, llm (Qwen), sandbox, forecast   (jaringan Docker, tidak terbuka ke internet)
```

## 1. Arahkan domain ke VPS

Di panel DNS domainmu, buat **A record**: nama `@` (atau subdomain seperti `analyst`), nilai `IP_VPS`.
Cek dari laptop sampai hasilnya IP VPS (bisa butuh beberapa menit sampai beberapa jam):

```bash
dig +short DOMAIN_KAMU
```

## 2. Siapkan server

```bash
ssh root@IP_VPS

apt update && apt upgrade -y
apt install -y git nginx certbot python3-certbot-nginx ufw
curl -fsSL https://get.docker.com | sh

# Firewall: hanya SSH dan web yang terbuka
ufw allow OpenSSH
ufw allow 'Nginx Full'
ufw --force enable
```

Jika VPS belum punya swap, tambahkan 4 GB agar build image tidak kehabisan memori:

```bash
swapon --show   # kosong = belum ada swap
fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

## 3. Ambil kode

```bash
mkdir -p /opt && cd /opt
git clone https://github.com/marsel23xxx/project_llm.git agentic-data-analyst
cd agentic-data-analyst
```

## 4. Buat `.env` production

```bash
cp .env.example .env
sed -i "s|^JWT_SECRET=.*|JWT_SECRET=$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')|" .env
sed -i "s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')|" .env
sed -i "s|^ENVIRONMENT=.*|ENVIRONMENT=production|" .env
sed -i "s|^FRONTEND_ORIGIN=.*|FRONTEND_ORIGIN=https://DOMAIN_KAMU|" .env
chmod 600 .env
```

`LLM_MODE=local` dibiarkan (biaya API nol). Jangan pernah mengubah `POSTGRES_PASSWORD` setelah database
pertama kali jalan: password lama sudah tersimpan di volume database.

## 5. Siapkan model

```bash
# LLM lokal (~2 GB)
./scripts/download_model.sh
```

Model IndoBERT (125 MB) tidak ada di git. Kirim dari **laptop** (terminal baru, bukan di dalam SSH):

```bash
scp ~/Downloads/sentiment_onnx.zip root@IP_VPS:/tmp/
```

Lalu di VPS:

```bash
apt install -y unzip
unzip -o /tmp/sentiment_onnx.zip -d backend/models/
ls backend/models/sentiment_onnx   # harus ada model.int8.onnx, tokenizer.json, config.joblib
```

Tanpa langkah ini aplikasi tetap jalan memakai baseline scikit-learn.

## 6. Jalankan

```bash
docker compose --profile llm up -d --build
```

Build pertama memakan 15 sampai 30 menit. Setelah selesai:

```bash
docker compose --profile llm ps                 # semua harus "Up"
curl -s localhost:8000/api/v1/health            # {"database":"ok","redis":"ok",...}
docker compose logs backend | grep "Model ML"   # sentimen=onnx:indobert-int8
```

## 7. Nginx dan HTTPS

```bash
sed "s/DOMAIN_KAMU/analyst.contoh.com/" deploy/nginx.conf > /etc/nginx/sites-available/agentic-data-analyst   # ganti domainnya
ln -sf /etc/nginx/sites-available/agentic-data-analyst /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx

certbot --nginx -d DOMAIN_KAMU
```

Certbot menanyakan email, lalu memasang sertifikat dan mengatur redirect ke HTTPS. Perpanjangan berjalan otomatis;
cek dengan `certbot renew --dry-run`.

## 8. Uji

Buka `https://DOMAIN_KAMU`, daftar akun, unggah `samples/penjualan_harian.csv`, lalu coba:

- "produk apa yang paling laku?"
- "perkiraan qty 14 hari ke depan"
- unggah `samples/ulasan_kedai_kopi.csv`, lalu "bagaimana sentimen ulasan per cabang?"

Jawaban pertama setelah server menganggur lama bisa lebih lambat karena model dimuat ulang ke memori.

## Perawatan

| Kebutuhan | Perintah (di `/opt/agentic-data-analyst`) |
| --- | --- |
| Update setelah `git push` | `git pull && docker compose --profile llm up -d --build` |
| Lihat log | `docker compose logs -f backend` |
| Pemakaian memori | `docker stats --no-stream` |
| Tutup pendaftaran akun | set `REGISTRATION_OPEN=false` di `.env`, lalu `docker compose up -d backend` |
| Backup database | `docker compose exec -T postgres pg_dump -U analyst analyst > backup_$(date +%F).sql` |
| Hentikan semua | `docker compose --profile llm down` (data tetap ada di volume) |
| Bersihkan image lama | `docker image prune -f` |

## Mengaktifkan mode hybrid (Claude API)

Isi di `.env`: `LLM_MODE=hybrid`, `ANTHROPIC_API_KEY=...`, dan `CLAUDE_MODEL` pilihanmu, lalu
`docker compose up -d backend`. Pasang spend limit di Claude Console sebelum mengaktifkannya.
