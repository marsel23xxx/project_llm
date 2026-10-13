"""Draw the charts of a written document as PNG images, for embedding in DOCX and PDF files.

The LLM describes a chart as data (see `parse_spec`); the drawing is done here so every chart
looks the same and never depends on the model's ability to draw.
"""

import io
import json
import math
import re
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

# Categorical palette in fixed order (validated light-surface steps); a chart never cycles past it.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID, SURFACE = "#171717", "#5f5e5a", "#e4e3df", "#ffffff"
MAX_BARS = 12
MAX_SLICES = len(PALETTE)
SCALE = 2  # drawn at twice the layout size so text stays sharp when printed
ESTIMATE_NOTE = "Angka perkiraan, perlu diverifikasi"
WEB_NOTE = "Sumber: hasil pencarian web, lihat Daftar Pustaka"
_ORIGINS = {"dokumen": "document", "document": "document", "file": "document", "sumber": "document", "web": "web"}


class ChartError(ValueError):
    pass


@dataclass
class ChartSpec:
    kind: str  # "bar" or "donut"
    title: str
    labels: list[str]
    values: list[float]
    unit: str = ""
    origin: str = "estimate"  # "document" (the user's attached files), "web" (search results) or "estimate"

    @property
    def from_source(self) -> bool:
        return self.origin != "estimate"

    @property
    def caption(self) -> str:
        return {"document": "Sumber: dokumen yang dilampirkan", "web": WEB_NOTE}.get(self.origin, ESTIMATE_NOTE)

    @property
    def reading(self) -> str:
        """What the chart shows, stated from the numbers themselves.

        The model's own paragraph about a chart can misread it (it once ranked a smaller bar above
        a larger one), so the figure carries a one-line reading that cannot disagree with the data.
        """
        pairs = sorted(zip(self.values, self.labels), reverse=True)
        suffix = f" {self.unit}" if self.unit and self.unit != "%" else ""
        if self.kind == "donut":
            total = sum(self.values)
            share = format_number(round(100 * pairs[0][0] / total, 1))
            return f"Bagian terbesar: {pairs[0][1]} ({share}%)."
        high, low = pairs[0], pairs[-1]
        percent = "%" if self.unit == "%" else ""
        return (
            f"Tertinggi: {high[1]} ({format_number(high[0])}{percent}{suffix}). "
            f"Terendah: {low[1]} ({format_number(low[0])}{percent}{suffix})."
        )


# Labels that already have an order of their own: years, dates, months, quarters, numbered steps.
_SEQUENCE_LABEL = re.compile(
    r"^\s*(?:\d[\d\s./-]*|(?:q|k|kuartal|triwulan|semester|tahap|fase|minggu|bulan|hari|tahun|periode|ke)[\s-]*\d+.*"
    r"|(?:jan|feb|mar|apr|mei|may|jun|jul|agu|aug|sep|okt|oct|nov|des|dec)\w*.*"
    r"|(?:senin|selasa|rabu|kamis|jumat|sabtu|minggu))\s*$",
    re.IGNORECASE,
)


def is_sequence(labels: list[str]) -> bool:
    return all(_SEQUENCE_LABEL.match(label) for label in labels)


def parse_spec(raw: str) -> ChartSpec:
    """Validate a chart block written by the LLM. Anything malformed is rejected, not guessed at."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ChartError("bukan JSON yang valid") from exc
    if not isinstance(data, dict):
        raise ChartError("harus berupa objek")

    kind = {"bar": "bar", "batang": "bar", "donut": "donut", "donat": "donut", "pie": "donut"}.get(str(data.get("type", "")).lower())
    labels, values = data.get("labels"), data.get("values")
    if kind is None or not isinstance(labels, list) or not isinstance(values, list):
        raise ChartError("butuh type, labels, dan values")
    if len(labels) != len(values) or not 2 <= len(labels) <= MAX_BARS:
        raise ChartError(f"jumlah label dan nilai harus sama, antara 2 dan {MAX_BARS}")
    try:
        numbers = [float(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise ChartError("values harus angka") from exc
    if any(math.isnan(v) or math.isinf(v) or v < 0 for v in numbers) or sum(numbers) <= 0:
        raise ChartError("nilai harus angka tidak negatif dan tidak semuanya nol")

    pairs = list(zip([str(label)[:40] for label in labels], numbers))
    if kind == "bar" and not is_sequence([label for label, _ in pairs]):
        # Categories are compared by size, so the largest comes first; time-like labels keep their order.
        pairs.sort(key=lambda p: -p[1])
    if kind == "donut" and len(pairs) > MAX_SLICES:
        # More slices than distinguishable colours: keep the largest and fold the rest together.
        pairs.sort(key=lambda p: -p[1])
        pairs = pairs[: MAX_SLICES - 1] + [("Lainnya", sum(v for _, v in pairs[MAX_SLICES - 1 :]))]
    return ChartSpec(
        kind=kind,
        title=str(data.get("title", "")).strip()[:120],
        labels=[label for label, _ in pairs],
        values=[value for _, value in pairs],
        unit=str(data.get("unit", "")).strip()[:20],
        origin=_ORIGINS.get(str(data.get("source", "")).lower(), "estimate"),
    )


def format_number(value: float) -> str:
    text = f"{value:,.0f}" if float(value).is_integer() or abs(value) >= 1000 else f"{value:,.2f}".rstrip("0").rstrip(".")
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def _font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size * SCALE)


def _width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    return int(draw.textlength(text, font=font))


def _fit(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
    if _width(draw, text, font) <= max_width:
        return text
    while len(text) > 1 and _width(draw, text + "…", font) > max_width:
        text = text[:-1]
    return text + "…"


def _frame(spec: ChartSpec, body_height: int) -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
    """A white canvas with the title on top and the provenance caption at the bottom; returns the body's top y."""
    # 600 layout units across a 14 cm figure puts 13-unit labels at about 8.5 pt on the page.
    width = 600 * SCALE
    top = (58 if spec.title else 24) * SCALE
    image = Image.new("RGB", (width, top + body_height + 64 * SCALE), SURFACE)
    draw = ImageDraw.Draw(image)
    if spec.title:
        draw.text((28 * SCALE, 20 * SCALE), _fit(draw, spec.title, _font(17), width - 56 * SCALE), fill=INK, font=_font(17))
    draw.text((28 * SCALE, image.height - 50 * SCALE), _fit(draw, spec.reading, _font(12), width - 56 * SCALE), fill=INK, font=_font(12))
    draw.text((28 * SCALE, image.height - 28 * SCALE), spec.caption, fill=MUTED, font=_font(11))
    return image, draw, top


def _bar(spec: ChartSpec) -> Image.Image:
    row, gap = 30 * SCALE, 12 * SCALE
    image, draw, top = _frame(spec, len(spec.values) * (row + gap))
    label_font, value_font = _font(13), _font(13)
    label_width = min(max(_width(draw, label, label_font) for label in spec.labels), 190 * SCALE)
    left = 28 * SCALE + label_width + 14 * SCALE
    suffix = f" {spec.unit}" if spec.unit else ""
    value_width = max(_width(draw, format_number(v) + suffix, value_font) for v in spec.values)
    track = image.width - left - value_width - 44 * SCALE
    peak = max(spec.values)

    draw.line([(left, top - 4 * SCALE), (left, top + len(spec.values) * (row + gap) - gap + 4 * SCALE)], fill=GRID, width=SCALE)
    for i, (label, value) in enumerate(zip(spec.labels, spec.values)):
        y = top + i * (row + gap)
        text = _fit(draw, label, label_font, label_width)
        draw.text((28 * SCALE + label_width - _width(draw, text, label_font), y + 6 * SCALE), text, fill=INK, font=label_font)
        length = max(int(track * value / peak), 2 * SCALE)
        draw.rounded_rectangle([left, y, left + length, y + row], radius=4 * SCALE, fill=PALETTE[0])
        draw.rectangle([left, y, left + 4 * SCALE, y + row], fill=PALETTE[0])  # square the end on the baseline
        draw.text((left + length + 10 * SCALE, y + 6 * SCALE), format_number(value) + suffix, fill=INK, font=value_font)
    return image


def _donut(spec: ChartSpec) -> Image.Image:
    size = 200 * SCALE
    image, draw, top = _frame(spec, max(size, len(spec.values) * 34 * SCALE) + 10 * SCALE)
    total = sum(spec.values)
    box = [30 * SCALE, top, 30 * SCALE + size, top + size]
    start = -90.0
    for i, value in enumerate(spec.values):
        sweep = 360 * value / total
        draw.pieslice(box, start, start + sweep, fill=PALETTE[i], outline=SURFACE, width=2 * SCALE)  # surface gap between slices
        start += sweep
    hole = size * 0.29
    draw.ellipse([box[0] + hole, box[1] + hole, box[2] - hole, box[3] - hole], fill=SURFACE)

    # Every slice is named beside a swatch with its value and share: identity never rests on colour alone.
    legend_x, font = box[2] + 30 * SCALE, _font(13)
    suffix = f" {spec.unit}" if spec.unit else ""
    available = image.width - legend_x - 28 * SCALE - 26 * SCALE
    for i, (label, value) in enumerate(zip(spec.labels, spec.values)):
        y = top + 8 * SCALE + i * 34 * SCALE
        draw.rounded_rectangle([legend_x, y + 3 * SCALE, legend_x + 14 * SCALE, y + 17 * SCALE], radius=3 * SCALE, fill=PALETTE[i])
        share = f"{format_number(round(100 * value / total, 1))}%"
        detail = share if spec.unit == "%" else f"{format_number(value)}{suffix} ({share})"
        text = _fit(draw, f"{label}: {detail}", font, available)
        draw.text((legend_x + 26 * SCALE, y), text, fill=INK, font=font)
    return image


def render_png(spec: ChartSpec) -> tuple[bytes, float]:
    """Return the PNG and its height/width ratio (for sizing it in a page)."""
    image = _bar(spec) if spec.kind == "bar" else _donut(spec)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue(), image.height / image.width
