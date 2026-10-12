"""Compare OCR, the vision model, and the combined policy on images of known text.

A printed "invoice" is rendered, then degraded the way real photos are (skew, blur, low resolution,
noise). For each image the script reports text accuracy and whether every number came out right.
Vision columns are filled only when the hosted API is configured (LLM_MODE=api + REMOTE_LLM_API_KEY).

    docker compose exec -T backend python - < ml/benchmark_ocr.py > ml/reports/ocr_vision.json

These are synthetic images. Real phone photos (perspective, shadows, handwriting) are harder.
"""

import difflib
import json
import random
import sys
import time

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from app.core.config import get_settings
from app.services import documents, ocr, vision

LINES = [
    "NOTA PEMBAYARAN LAUNDRY", "Kopi Nusantara - Cabang Bandung", "Nomor nota : INV-2026-0142", "Periode : September 2026",
    "Linen biasa : 2.150 kg x Rp 4.500", "Linen infeksius : 310 kg x Rp 7.000", "Total tagihan : Rp 11.845.000",
    "Jatuh tempo : 10 Oktober 2026", "Denda keterlambatan 2 persen per bulan dari nilai tagihan",
]  # fmt: skip
TRUTH = " ".join(LINES)
NUMBERS = ["11.845.000", "INV-2026-0142", "2.150", "4.500", "7.000"]


def render() -> Image.Image:
    image = Image.new("RGB", (1500, 120 + 76 * len(LINES)), "white")
    draw, font = ImageDraw.Draw(image), ImageFont.load_default(size=40)
    for i, line in enumerate(LINES):
        draw.text((70, 60 + 76 * i), line, fill=(20, 20, 20), font=font)
    return image


def noisy(image: Image.Image) -> Image.Image:
    out, rng = image.copy(), random.Random(1)
    pixels = out.load()
    for _ in range(60000):
        pixels[rng.randrange(out.width), rng.randrange(out.height)] = (0, 0, 0)
    return out


def cases() -> dict[str, Image.Image]:
    base = render()
    rotate = lambda deg: base.rotate(deg, expand=True, fillcolor="white")  # noqa: E731
    return {
        "bersih": base,
        "miring 4 derajat": rotate(4),
        "miring 8 derajat": rotate(8),
        "miring 15 derajat": rotate(15),
        "buram ringan": base.filter(ImageFilter.GaussianBlur(3)),
        "buram berat": base.filter(ImageFilter.GaussianBlur(5)),
        "resolusi rendah (300 px)": base.resize((300, int(base.height * 300 / base.width))),
        "gelap, kontras rendah": Image.eval(base, lambda v: int(v * 0.35) + 40),
        "berbintik": noisy(base),
    }


def score(text: str) -> dict:
    flat = " ".join(text.split())
    return {
        "accuracy": round(difflib.SequenceMatcher(None, TRUTH.lower(), flat.lower()).ratio(), 3),
        "numbers_correct": all(n in flat for n in NUMBERS),
    }


def main() -> None:
    reader = vision.reader_from(get_settings())
    rows = []
    for name, image in cases().items():
        result = ocr.read(image)
        row = {"image": name, "ocr": {**score(result.text), "confidence": round(result.confidence, 1), "words": result.words}}
        if reader:
            start = time.perf_counter()
            try:
                row["vision"] = {**score(reader(image)), "seconds": round(time.perf_counter() - start, 1)}
            except vision.VisionError as exc:
                row["vision"] = {"error": str(exc)[:120]}
        text, method = documents.read_image(image, reader)
        row["combined"] = {**score(text), "read_by": method}
        rows.append(row)
        print(f"{name:26s} OCR {row['ocr']}  gabungan {row['combined']}  vision {row.get('vision', '-')}", file=sys.stderr)

    def tally(key: str) -> str:
        scored = [r[key] for r in rows if "numbers_correct" in r.get(key, {})]
        return f"{sum(s['numbers_correct'] for s in scored)}/{len(scored)}" if scored else "tidak diukur"

    report = {
        "note": "Gambar sintetis dari teks cetak; foto HP sungguhan lebih sulit.",
        "ocr_engine": f"tesseract ({ocr.LANGUAGES})",
        "vision_model": reader.name if reader else None,
        "ocr_min_confidence": documents.OCR_MIN_CONFIDENCE,
        "images_with_all_numbers_correct": {"ocr": tally("ocr"), "vision": tally("vision"), "combined": tally("combined")},
        "sent_to_vision": sum(r["combined"]["read_by"] == "vision" for r in rows),
        "rows": rows,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


main()
