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
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from app.ml.text import normalize

MAX_PAGES = 300
MIN_CHARS_PER_PAGE = 30  # below this on average, the PDF is almost certainly scanned images
# A CPU-only LLM reads roughly 15-20 prompt tokens per second, so prompt size is the latency:
# small chunks and few of them keep a document answer under a minute.
CHUNK_WORDS = 110
OVERLAP_WORDS = 20
TOP_K = 3
SUMMARY_EXCERPTS = 12
SUMMARY_EXCERPT_WORDS = 70


class DocumentError(ValueError):
    pass


@dataclass
class Chunk:
    page: int
    position: int
    text: str


def extract_pages(path: Path) -> list[str]:
    try:
        with pdfplumber.open(path) as pdf:
            if len(pdf.pages) > MAX_PAGES:
                raise DocumentError(f"PDF memiliki {len(pdf.pages)} halaman; batasnya {MAX_PAGES} halaman.")
            pages = [(page.extract_text() or "").strip() for page in pdf.pages]
    except DocumentError:
        raise
    except Exception as exc:  # pdfminer raises many parser-specific errors
        raise DocumentError("File PDF tidak bisa dibaca. Pastikan file tidak rusak atau terkunci password.") from exc

    if not pages:
        raise DocumentError("PDF tidak memiliki halaman.")
    if sum(len(p) for p in pages) < MIN_CHARS_PER_PAGE * len(pages):
        raise DocumentError(
            "PDF ini tampaknya hasil scan atau foto, sehingga teksnya tidak bisa dibaca. "
            "Saat ini hanya PDF berisi teks asli (bisa diblok dan disalin) yang didukung."
        )
    return pages


def chunk_pages(pages: list[str]) -> list[Chunk]:
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
            chunks.append(Chunk(page=page_number, position=len(chunks), text=" ".join(window)))
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


def summary_excerpts(chunks: list[Chunk]) -> tuple[list[Chunk], bool]:
    """Material for a whole-document summary within a small prompt budget.

    A short document is used in full. A long one is sampled: the opening of evenly spaced chunks,
    so the summary reflects the whole document instead of only its first pages.
    """
    if len(chunks) <= SUMMARY_EXCERPTS:
        return chunks, False
    step = (len(chunks) - 1) / (SUMMARY_EXCERPTS - 1)
    picked = [chunks[round(i * step)] for i in range(SUMMARY_EXCERPTS)]
    return [Chunk(c.page, c.position, " ".join(c.text.split()[:SUMMARY_EXCERPT_WORDS])) for c in picked], True
