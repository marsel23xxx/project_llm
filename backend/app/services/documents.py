"""PDF documents: text extraction, chunking, and retrieval of the passages relevant to a question.

Retrieval is lexical (TF-IDF over words and character n-grams, cosine similarity). It needs no extra
model, is fast for a single document, and tolerates typos and Indonesian affixes through the
character n-grams. Dense embeddings can replace `Retriever` later without touching its callers.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pdfplumber
from PIL import Image, UnidentifiedImageError
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from app.ml.text import normalize
from app.services import ocr

MAX_PAGES = 300
MIN_CHARS_PER_PAGE = 30  # a page with less text than this is treated as a scanned image
# OCR takes a few seconds per page on CPU and uploads are processed while the user waits.
MAX_OCR_PAGES = 30
OCR_DPI = 200
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
# A CPU-only LLM reads roughly 15-20 prompt tokens per second, so prompt size is the latency:
# small chunks and few of them keep a document answer under a minute.
CHUNK_WORDS = 110
OVERLAP_WORDS = 20
TOP_K = 3
# A hosted model reads a long prompt in a second or two, so it can be given more of the document.
TOP_K_HOSTED = 6
SUMMARY_EXCERPTS = 12
SUMMARY_EXCERPTS_HOSTED = 40
SUMMARY_EXCERPT_WORDS = 70


class DocumentError(ValueError):
    pass


@dataclass
class Chunk:
    page: int
    position: int
    text: str
    source: str = ""  # file name, when a session holds several documents


@dataclass
class Extraction:
    pages: list[str]
    ocr_pages: int = 0  # how many pages were read by OCR instead of from embedded text


def extract_pages(path: Path) -> Extraction:
    """Text of each PDF page. Pages without embedded text (scans) are read with OCR."""
    ocr_pages = 0
    try:
        with pdfplumber.open(path) as pdf:
            if len(pdf.pages) > MAX_PAGES:
                raise DocumentError(f"PDF memiliki {len(pdf.pages)} halaman; batasnya {MAX_PAGES} halaman.")
            pages = []
            for page in pdf.pages:
                text = (page.extract_text() or "").strip()
                if len(text) < MIN_CHARS_PER_PAGE and ocr.available():
                    if ocr_pages >= MAX_OCR_PAGES:
                        raise DocumentError(
                            f"PDF ini berisi lebih dari {MAX_OCR_PAGES} halaman hasil scan. "
                            f"Pecah menjadi file yang lebih kecil agar bisa dibaca dengan OCR."
                        )
                    scanned = ocr.read_text(page.to_image(resolution=OCR_DPI).original)
                    if len(scanned) > len(text):
                        text, ocr_pages = scanned, ocr_pages + 1
                pages.append(text)
    except DocumentError:
        raise
    except Exception as exc:  # pdfminer raises many parser-specific errors
        raise DocumentError("File PDF tidak bisa dibaca. Pastikan file tidak rusak atau terkunci password.") from exc

    if not pages:
        raise DocumentError("PDF tidak memiliki halaman.")
    if sum(len(p) for p in pages) < MIN_CHARS_PER_PAGE * len(pages):
        if not ocr.available():
            raise DocumentError("PDF ini hasil scan dan mesin OCR tidak tersedia di server, sehingga teksnya tidak bisa dibaca.")
        raise DocumentError(
            "Teks di PDF hasil scan ini tidak terbaca oleh OCR. Coba scan ulang dengan resolusi lebih tinggi dan posisi lurus."
        )
    return Extraction(pages, ocr_pages)


def extract_image(path: Path) -> Extraction:
    """Text of a photo or screenshot, read with OCR, as a one-page document."""
    try:
        with Image.open(path) as image:
            image.load()
            text = ocr.read_text(image)
    except ocr.OcrUnavailable as exc:
        raise DocumentError(str(exc)) from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise DocumentError("File gambar tidak bisa dibuka. Gunakan PNG, JPG, atau WEBP.") from exc
    if len(text) < MIN_CHARS_PER_PAGE:
        raise DocumentError(
            "Tidak ada teks yang terbaca dari gambar ini. OCR bekerja baik untuk foto dokumen yang lurus, terang, dan tajam."
        )
    return Extraction([text], ocr_pages=1)


def chunk_pages(pages: list[str], source: str = "") -> list[Chunk]:
    """Split each page into overlapping word windows; a chunk never spans pages so it can cite one."""
    chunks: list[Chunk] = []
    for page_number, text in enumerate(pages, start=1):
        words = re.sub(r"[ \t]+", " ", text).split()
        if not words:
            continue
        step = CHUNK_WORDS - OVERLAP_WORDS
        for start in range(0, len(words), step):
            window = words[start : start + CHUNK_WORDS]
            # A short tail is already covered by the previous window's overlap.
            if start > 0 and len(window) <= OVERLAP_WORDS:
                break
            chunks.append(Chunk(page=page_number, position=len(chunks), text=" ".join(window), source=source))
    return chunks


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        texts = [normalize(c.text) for c in chunks]
        self._word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self._char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self._matrix = hstack([self._word.fit_transform(texts), self._char.fit_transform(texts)]).tocsr()

    def search(self, question: str, k: int = TOP_K) -> list[tuple[Chunk, float]]:
        query = normalize(question)
        vector = hstack([self._word.transform([query]), self._char.transform([query])]).tocsr()
        scores = np.asarray((self._matrix @ vector.T).todense()).ravel()
        norms = np.sqrt(self._matrix.multiply(self._matrix).sum(axis=1)).A1 * np.sqrt(vector.multiply(vector).sum())
        scores = np.divide(scores, norms, out=np.zeros_like(scores), where=norms > 0)
        best = [int(i) for i in np.argsort(-scores)[:k] if scores[i] > 0]
        # Weakest first, strongest last: a small model leans on the text nearest the question.
        return [(self.chunks[i], float(scores[i])) for i in reversed(best)]


def summary_excerpts(chunks: list[Chunk], limit: int = SUMMARY_EXCERPTS) -> tuple[list[Chunk], bool]:
    """Material for a whole-document summary within a small prompt budget.

    A short document is used in full. A long one is sampled: the opening of evenly spaced chunks,
    so the summary reflects the whole document instead of only its first pages.
    """
    if len(chunks) <= limit:
        return chunks, False
    step = (len(chunks) - 1) / (limit - 1)
    picked = [chunks[round(i * step)] for i in range(limit)]
    return [Chunk(c.page, c.position, " ".join(c.text.split()[:SUMMARY_EXCERPT_WORDS]), c.source) for c in picked], True


def page_texts(chunks: list[Chunk]) -> list[tuple[str, int, str]]:
    """Rebuild each page's text as (file, page, text), dropping the words repeated by the overlap."""
    pages: dict[tuple[str, int], list[str]] = {}
    for chunk in sorted(chunks, key=lambda c: c.position):
        words = chunk.text.split()
        key = (chunk.source, chunk.page)
        if key in pages:
            words = words[OVERLAP_WORDS:]
        pages.setdefault(key, []).extend(words)
    return [(source, page, " ".join(words)) for (source, page), words in pages.items()]


def word_count(chunks: list[Chunk]) -> int:
    return sum(len(text.split()) for _, _, text in page_texts(chunks))


def renumber(groups: list[list[Chunk]]) -> list[Chunk]:
    """Join the chunks of several files into one corpus with a single running position."""
    merged: list[Chunk] = []
    for group in groups:
        for chunk in sorted(group, key=lambda c: c.position):
            merged.append(Chunk(chunk.page, len(merged), chunk.text, chunk.source))
    return merged
