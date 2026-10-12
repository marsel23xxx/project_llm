"""OCR for photos and scanned pages, using the Tesseract engine installed in the backend image.

It runs on the server, so images never leave it. Clean scans and straight photos of printed text
read well; handwriting, skewed or blurry photos, and low-resolution screenshots do not.
"""

import shutil
from dataclasses import dataclass

from PIL import Image, ImageOps

LANGUAGES = "ind+eng"
MIN_SIDE = 1400  # Tesseract is tuned for ~300 dpi; small images are enlarged to about this size
MAX_SIDE = 4000


class OcrUnavailable(RuntimeError):
    pass


def available() -> bool:
    return shutil.which("tesseract") is not None


def prepare(image: Image.Image) -> Image.Image:
    """Normalise orientation, colour and size the way Tesseract reads best."""
    image = ImageOps.exif_transpose(image)  # phone photos store rotation in metadata
    image = ImageOps.autocontrast(image.convert("L"))
    longest = max(image.size)
    if longest < MIN_SIDE:
        scale = MIN_SIDE / longest
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    elif longest > MAX_SIDE:
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return image


@dataclass
class OcrResult:
    text: str
    words: int
    # Mean per-word confidence (0-100). On test images, 90+ meant every number was read correctly,
    # around 80 meant some were wrong, and below that the text was unusable.
    confidence: float


def read(image: Image.Image) -> OcrResult:
    if not available():
        raise OcrUnavailable("Mesin OCR tidak terpasang di server ini.")
    import pytesseract

    prepared = prepare(image)
    data = pytesseract.image_to_data(prepared, lang=LANGUAGES, output_type=pytesseract.Output.DICT)
    scores = [float(c) for word, c in zip(data["text"], data["conf"]) if word.strip() and float(c) >= 0]
    text = pytesseract.image_to_string(prepared, lang=LANGUAGES).strip()
    return OcrResult(text=text, words=len(scores), confidence=sum(scores) / len(scores) if scores else 0.0)


def read_text(image: Image.Image) -> str:
    return read(image).text
