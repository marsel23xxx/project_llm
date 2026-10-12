import json

import httpx
import pytest
from PIL import Image

from app.core.config import Settings
from app.services import documents, ocr, vision


def _ocr(monkeypatch, text: str, words: int, confidence: float):
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "read", lambda image: ocr.OcrResult(text, words, confidence))


IMAGE = Image.new("RGB", (200, 100), "white")


def test_confident_ocr_is_used_and_the_image_never_leaves_the_server(monkeypatch):
    _ocr(monkeypatch, "Total tagihan Rp 11.845.000", words=4, confidence=93.0)
    calls = []
    assert documents.read_image(IMAGE, lambda image: calls.append(1) or "x") == ("Total tagihan Rp 11.845.000", "ocr")
    assert calls == []


@pytest.mark.parametrize(("words", "confidence"), [(40, 80.0), (40, 32.9), (0, 0.0), (2, 95.0)])
def test_unsure_ocr_hands_the_image_to_vision(monkeypatch, words, confidence):
    _ocr(monkeypatch, "T0tal tag1han", words, confidence)
    assert documents.read_image(IMAGE, lambda image: "Total tagihan Rp 11.845.000") == ("Total tagihan Rp 11.845.000", "vision")


def test_unsure_ocr_is_kept_when_there_is_no_vision_or_it_fails(monkeypatch):
    _ocr(monkeypatch, "T0tal tag1han", words=40, confidence=60.0)
    assert documents.read_image(IMAGE, None) == ("T0tal tag1han", "ocr")

    def broken(image):
        raise vision.VisionError("saldo habis")

    assert documents.read_image(IMAGE, broken) == ("T0tal tag1han", "ocr")
    assert documents.read_image(IMAGE, lambda image: "   ") == ("T0tal tag1han", "ocr")


def test_image_extraction_records_how_it_was_read(monkeypatch, tmp_path):
    path = tmp_path / "foto.png"
    IMAGE.save(path)
    _ocr(monkeypatch, "", words=0, confidence=0.0)
    extraction = documents.extract_image(path, lambda image: "Deskripsi gambar: grafik batang penjualan per bulan, tertinggi di Juli.")
    assert (extraction.ocr_pages, extraction.vision_pages) == (0, 1) and "grafik batang" in extraction.pages[0]
    with pytest.raises(documents.DocumentError):  # nothing readable and no vision model to ask
        documents.extract_image(path, None)


def test_vision_reader_is_only_built_when_the_hosted_api_is_in_use():
    assert vision.reader_from(Settings(llm_mode="local", remote_llm_api_key="sk-x")) is None
    assert vision.reader_from(Settings(llm_mode="api", remote_llm_api_key=None)) is None
    assert vision.reader_from(Settings(llm_mode="api", remote_llm_api_key="sk-x", vision_fallback=False)) is None
    assert vision.reader_from(Settings(llm_mode="api", remote_llm_api_key="sk-x")).name == "vision:deepseek-flash"


def test_vision_request_sends_one_downsized_jpeg_and_returns_the_text():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"], seen["body"] = str(request.url), request.headers["authorization"], json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": " Nota 042 \n"}}]})

    reader = vision.VisionReader("https://api.deepseek.com/", "sk-x", "deepseek-flash")
    reader._client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.deepseek.com", headers=reader._client.headers)
    assert reader(Image.new("RGB", (4000, 3000), "white")) == "Nota 042"

    content = seen["body"]["messages"][0]["content"]
    assert seen["url"] == "https://api.deepseek.com/chat/completions" and seen["auth"] == "Bearer sk-x"
    assert seen["body"]["messages"][0]["role"] == "user" and seen["body"]["thinking"] == {"type": "disabled"}
    assert content[0]["type"] == "text" and "[tidak terbaca]" in content[0]["text"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,") and content[1]["image_url"]["detail"] == "high"
    assert len(content[1]["image_url"]["url"]) < 200_000  # 4000x3000 was reduced before sending

    reader._client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(402, text="Insufficient Balance")), base_url="https://api.deepseek.com")
    with pytest.raises(vision.VisionError, match="402"):
        reader(IMAGE)
