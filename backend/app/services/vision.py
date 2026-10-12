"""Read an image with a hosted vision model, for the pictures OCR cannot read reliably.

Used only when OCR is unsure (skewed or blurry photos, handwriting, charts) and a hosted
OpenAI-compatible API key is configured. Clean scans never leave the server.
"""

import base64
import io

import httpx
from PIL import Image, ImageOps

from app.core.config import Settings

MAX_SIDE = 1600  # the provider downsizes to about 1300x1300 anyway; sending more only costs bandwidth
PROMPT = (
    "Salin seluruh teks pada gambar ini persis seperti tertulis, dengan urutan baca dan pemisahan baris yang sama. "
    "Tabel ditulis satu baris per baris tabel dengan pemisah \" | \". "
    "Jangan menambah, menerjemahkan, merangkum, atau memperbaiki isi. Angka disalin apa adanya; "
    "bagian yang tidak terbaca ditulis [tidak terbaca], jangan ditebak. "
    "Jika ada grafik, diagram, atau foto, tambahkan di akhir satu paragraf yang diawali \"Deskripsi gambar:\" "
    "yang menjelaskannya secara singkat. Jangan menulis apa pun selain itu."
)


class VisionError(RuntimeError):
    pass


class VisionReader:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0):
        self._model = model
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, headers={"Authorization": f"Bearer {api_key}"})
        self.name = f"vision:{model}"

    def __call__(self, image: Image.Image) -> str:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88)
        data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
        payload = {
            "model": self._model,
            "max_tokens": 2500,
            "temperature": 0,
            "thinking": {"type": "disabled"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                    ],
                }
            ],
        }
        try:
            response = self._client.post("/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise VisionError(f"API vision tidak bisa dihubungi: {exc}") from exc
        if response.status_code >= 400:
            raise VisionError(f"API vision error {response.status_code}: {response.text[:200]}")
        text = (response.json()["choices"][0]["message"].get("content") or "").strip()
        if not text:
            raise VisionError("API vision tidak mengembalikan teks")
        return text


def reader_from(settings: Settings) -> VisionReader | None:
    """A vision reader when the hosted OpenAI-compatible API is in use and vision is allowed; else None."""
    if settings.llm_mode == "local" or not settings.remote_llm_api_key or not settings.vision_fallback:
        return None
    return VisionReader(settings.remote_llm_base_url, settings.remote_llm_api_key, settings.vision_model)
