"""OCR for photos and scanned pages, using the Tesseract engine installed in the backend image.

It runs on the server, so images never leave it. Clean scans and straight photos of printed text
read well; handwriting, skewed or blurry photos, and low-resolution screenshots do not.
"""

import shutil

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


def read_text(image: Image.Image) -> str:
    if not available():
        raise OcrUnavailable("Mesin OCR tidak terpasang di server ini.")
    import pytesseract

    return pytesseract.image_to_string(prepare(image), lang=LANGUAGES).strip()
