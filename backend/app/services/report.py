"""Render one stored agent answer as a PDF report (question, answer, chart, table, method).

Everything comes from the saved agent events, so the PDF shows exactly what the chat showed.
"""

import io
import re
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

MAX_TABLE_ROWS = 40
MAX_TABLE_COLS = 7
MAX_CODE_CHARS = 3000
SERIES_COLORS = [colors.HexColor("#2a78d6"), colors.HexColor("#eb6834"), colors.HexColor("#1baf7a")]
INK = colors.HexColor("#171717")
MUTED = colors.HexColor("#5f5e5a")
GRID = colors.HexColor("#e4e3df")
INTENT_LABEL = {
    "sql": "Query SQL",
    "python": "Analisis Python",
    "sentiment": "Analisis sentimen",
    "forecast": "Prediksi",
    "profile": "Ringkasan dataset",
    "doc_qa": "Tanya dokumen",
    "doc_summary": "Ringkasan dokumen",
    "doc_analysis": "Telaah dokumen",
    "chat": "Obrolan",
}

_styles = getSampleStyleSheet()
TITLE = ParagraphStyle("title", parent=_styles["Title"], fontSize=17, leading=21, alignment=0, textColor=INK, spaceAfter=2)
META = ParagraphStyle("meta", parent=_styles["Normal"], fontSize=8.5, textColor=MUTED, spaceAfter=10)
HEADING = ParagraphStyle("heading", parent=_styles["Heading3"], fontSize=11, textColor=INK, spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("body", parent=_styles["Normal"], fontSize=10, leading=14.5, textColor=INK)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.5, leading=11.5, textColor=MUTED)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8, leading=10)
CODE = ParagraphStyle("code", parent=_styles["Code"], fontSize=7.5, leading=9.5, textColor=INK, backColor=colors.HexColor("#f4f4f2"))


def _fmt(value, type_name: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        text = f"{value:,.2f}".rstrip("0").rstrip(".") if isinstance(value, float) and not value.is_integer() else f"{int(value):,}"
        return text.replace(",", "_").replace(".", ",").replace("_", ".")
    text = str(value)
    if any(k in type_name.upper() for k in ("TIMESTAMP", "DATE")):
        text = text[:10]
    return text


def _paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(text).replace("\n", "<br/>"), style)


_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CODE_SPAN = re.compile(r"`([^`]+)`")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_HEADING = re.compile(r"^#{1,6}\s+")
BULLET_STYLE = ParagraphStyle("bullet", parent=BODY, leftIndent=12, bulletIndent=2)


def _inline(text: str) -> str:
    text = escape(text)
    text = _BOLD.sub(r"<b>\1</b>", text)
    return _CODE_SPAN.sub(r'<font face="Courier">\1</font>', text)


def markdown_flowables(text: str) -> list:
    """Render the light Markdown LLM answers use (bold, lists, headings) instead of printing the marks."""
    flow = []
    for line in text.splitlines():
        if not line.strip():
            flow.append(Spacer(1, 4))
        elif _HEADING.match(line):
            flow.append(Paragraph(f"<b>{_inline(_HEADING.sub('', line))}</b>", BODY))
        elif _BULLET.match(line):
            marker = _BULLET.match(line).group().strip()
            flow.append(Paragraph(_inline(_BULLET.sub("", line)), BULLET_STYLE, bulletText="•" if marker in "-*•" else marker))
        else:
            flow.append(Paragraph(_inline(line), BODY))
    return flow


def _table(table: dict) -> list:
    columns = table["columns"][:MAX_TABLE_COLS]
    types = table["types"][:MAX_TABLE_COLS]
    # A forecast table lists history first and the predictions last; the predictions are the point.
    keep_tail = "prediksi" in table["columns"]
    rows = table["rows"][-MAX_TABLE_ROWS:] if keep_tail else table["rows"][:MAX_TABLE_ROWS]
    numeric = [any(k in t.upper() for k in ("INT", "DOUBLE", "FLOAT", "DECIMAL")) for t in types]
    data = [
        [_paragraph(c, ParagraphStyle("h", parent=CELL, textColor=MUTED, alignment=2 if num else 0)) for c, num in zip(columns, numeric)]
    ]
    for row in rows:
        data.append(
            [
                _paragraph(_fmt(v, t)[:80], ParagraphStyle("r", parent=CELL, alignment=2) if num else CELL)
                for v, t, num in zip(row[:MAX_TABLE_COLS], types, numeric)
            ]
        )
    width = (A4[0] - 36 * mm) / len(columns)
    flow = Table(data, colWidths=[width] * len(columns), repeatRows=1)
    flow.setStyle(
        TableStyle(
            [
                ("LINEBELOW", (0, 0), (-1, 0), 0.8, MUTED),
                ("LINEBELOW", (0, 1), (-1, -1), 0.3, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    out = [flow]
    hidden_rows = len(table["rows"]) - len(rows)
    hidden_cols = len(table["columns"]) - len(columns)
    notes = []
    if hidden_rows > 0 or table.get("truncated"):
        which = "terakhir " if keep_tail else ""
        notes.append(f"menampilkan {len(rows)} baris {which}dari {len(table['rows'])}{'+' if table.get('truncated') else ''}")
    if hidden_cols > 0:
        notes.append(f"{hidden_cols} kolom terakhir tidak ditampilkan")
    if notes:
        out.append(_paragraph("Catatan: " + "; ".join(notes) + ".", SMALL))
    return out


def _chart(spec: dict, table: dict) -> Drawing | None:
    columns = table["columns"]
    if spec["x"] not in columns or not all(y in columns for y in spec["y"]):
        return None
    xi = columns.index(spec["x"])
    x_type = table["types"][xi]
    labels = [_fmt(row[xi], x_type)[:14] for row in table["rows"]]
    series = [[row[columns.index(y)] for row in table["rows"]] for y in spec["y"]]
    if not labels:
        return None

    width, height = A4[0] - 36 * mm, 190
    drawing = Drawing(width, height)
    plot_x, plot_y, plot_w, plot_h = 46, 46, width - 60, height - 70
    step = max(1, len(labels) // 8)

    if spec["kind"] == "bar":
        chart = VerticalBarChart()
        chart.data = [[v if isinstance(v, (int, float)) else 0 for v in s] for s in series]
        chart.categoryAxis.categoryNames = [lab if i % step == 0 else "" for i, lab in enumerate(labels)]
        chart.categoryAxis.labels.fontSize = 7
        chart.categoryAxis.strokeColor = GRID
        chart.valueAxis.valueMin = 0
        chart.barSpacing = 1
        for i in range(len(series)):
            chart.bars[i].fillColor = SERIES_COLORS[i % 3]
            chart.bars[i].strokeColor = None
        axis = chart.valueAxis
    else:
        chart = LinePlot()
        # Points with no value (e.g. the forecast part of the "actual" line) are simply left out.
        chart.data = [[(i, v) for i, v in enumerate(s) if isinstance(v, (int, float))] or [(0, 0)] for s in series]
        chart.xValueAxis.valueMin, chart.xValueAxis.valueMax = 0, max(1, len(labels) - 1)
        chart.xValueAxis.valueSteps = list(range(0, len(labels), step))
        chart.xValueAxis.labelTextFormat = lambda v: labels[int(v)] if 0 <= int(v) < len(labels) else ""
        chart.xValueAxis.labels.fontSize = 7
        chart.xValueAxis.strokeColor = GRID
        dashed = set(spec.get("dashed", []))
        for i, name in enumerate(spec["y"]):
            chart.lines[i].strokeColor = SERIES_COLORS[i % 3]
            chart.lines[i].strokeWidth = 1.6
            if name in dashed:
                chart.lines[i].strokeDashArray = (4, 3)
        axis = chart.yValueAxis

    chart.x, chart.y, chart.width, chart.height = plot_x, plot_y, plot_w, plot_h
    axis.labels.fontSize = 7
    axis.strokeColor = None
    axis.visibleGrid = True
    axis.gridStrokeColor = GRID
    axis.gridStrokeWidth = 0.4
    axis.labelTextFormat = lambda v: _fmt(float(v))
    drawing.add(chart)

    # Legend: a colour swatch beside each series name, names in text ink.
    cursor = plot_x
    for i, name in enumerate(spec["y"]):
        if spec["kind"] == "bar":
            drawing.add(Rect(cursor, 12, 9, 7, fillColor=SERIES_COLORS[i % 3], strokeColor=None))
        else:
            dash = (4, 3) if name in set(spec.get("dashed", [])) else None
            drawing.add(Line(cursor, 15.5, cursor + 14, 15.5, strokeColor=SERIES_COLORS[i % 3], strokeWidth=1.6, strokeDashArray=dash))
        drawing.add(String(cursor + 18, 12, name, fontSize=8, fillColor=MUTED))
        cursor += 30 + 4.6 * len(name)
    return drawing


def build_pdf(question: str, answer: str, steps: list[dict], dataset_name: str, created_at: datetime) -> bytes:
    def last(kind: str) -> dict | None:
        return next((s for s in reversed(steps) if s.get("type") == kind), None)

    intent, table, chart = last("intent"), last("table"), last("chart")
    samples, note, code, sources = last("samples"), last("stdout"), last("code"), last("sources")

    story = [
        _paragraph("Laporan Analisis Data", TITLE),
        _paragraph(
            f"Dataset: {dataset_name}  |  {created_at.strftime('%d-%m-%Y %H:%M')} UTC"
            + (f"  |  Metode: {INTENT_LABEL.get(intent['value'], intent['value'])}" if intent else ""),
            META,
        ),
        _paragraph("Pertanyaan", HEADING),
        _paragraph(question, BODY),
        _paragraph("Jawaban", HEADING),
        *markdown_flowables(answer or "-"),
    ]
    if chart and table:
        drawing = _chart(chart["spec"], table)
        if drawing:
            story += [_paragraph("Grafik", HEADING), drawing]
    if table:
        story += [_paragraph("Data", HEADING), *_table(table)]
    if samples and samples.get("items"):
        story.append(_paragraph("Contoh teks", HEADING))
        story += [_paragraph(f"[{item['label']}] {item['text']}", SMALL) for item in samples["items"]]
    if sources and sources.get("items"):
        story.append(_paragraph("Sumber di dokumen", HEADING))
        story += [_paragraph(f"Hal. {item['page']}: {item['text']}", SMALL) for item in sources["items"]]
    if note or code:
        story.append(_paragraph("Cara perhitungan", HEADING))
        if note:
            story.append(_paragraph(note["value"], SMALL))
        if code:
            story += [Spacer(1, 4), Preformatted(code["value"][:MAX_CODE_CHARS], CODE)]
    story += [Spacer(1, 14), _paragraph("Dibuat otomatis oleh Agentic Data Analyst. Periksa kembali angka penting sebelum dipakai untuk keputusan.", SMALL)]

    buffer = io.BytesIO()
    SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
        title="Laporan Analisis Data", author="Agentic Data Analyst",
    ).build(story)  # fmt: skip
    return buffer.getvalue()
