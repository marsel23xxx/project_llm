"""Keep a written document honest about where its facts come from.

The model writes the prose and cites sources as [n]; everything that can be checked is done here in
code instead of being trusted: the reference list is built from the sources that were really cited,
and a chart may only claim a source when its numbers are actually found in that source's text.
"""

import json
import re
from datetime import date

from app.services.websearch import WebSource

_CHART_BLOCK = re.compile(r"(```chart\s*\n.*?\n\s*```)", re.DOTALL)
_CITATION = re.compile(r"\[(\d{1,2}(?:\s*[,;]\s*\d{1,2})*)\]")
_REFERENCES_HEADING = re.compile(r"^#{1,3}\s*(daftar pustaka|referensi|rujukan|sumber)\s*$.*", re.IGNORECASE | re.MULTILINE | re.DOTALL)
MONTHS = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]


def long_date(day: date) -> str:
    return f"{day.day} {MONTHS[day.month - 1]} {day.year}"


def parse_queries(text: str, fallback: str, limit: int = 3) -> list[str]:
    """Search queries, one per line, as written by the model; the user's own words if it wrote none."""
    queries = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip("\"'`")
        if 3 <= len(line) <= 200 and line.lower() not in (q.lower() for q in queries):
            queries.append(line)
    return queries[:limit] or [fallback[:200]]


def _number_forms(value: float) -> list[str]:
    if float(value).is_integer():
        n = int(value)
        return list({str(n), f"{n:,}", f"{n:,}".replace(",", ".")})
    text = f"{value:.4f}".rstrip("0")
    return [text, text.replace(".", ",")]


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def _numbers_in(text: str) -> set[float]:
    """Every number written in the text, read both ways: "38.310,89" (Indonesian) and "38,310.89" (English)."""
    found: set[float] = set()
    for token in _NUMBER.findall(text):
        for reading in (token.replace(".", "").replace(",", "."), token.replace(",", "")):
            try:
                found.add(float(reading))
            except ValueError:
                pass
    return found


def _decimals(value: float) -> int:
    text = f"{value:.4f}".rstrip("0")
    return len(text) - text.index(".") - 1


# A chart in "juta ha" may be drawn from a source that says "202.004,30 ha": the same figure in another unit.
_UNIT_SCALES = {"ribu": 1e3, "juta": 1e6, "miliar": 1e9, "milyar": 1e9, "triliun": 1e12}


def number_in_text(value: float, text: str, numbers: set[float] | None = None, unit: str = "") -> bool:
    """True when the number is written in the text, in either digit grouping, or is one of its numbers rounded.

    Rounding counts because a chart of "38.311 ha" drawn from "38.310,89 ha" is still the source's figure.
    """
    if any(re.search(rf"(?<![\d.,]){re.escape(form)}(?![.,]?\d)", text) for form in _number_forms(value)):
        return True
    scale = 10 ** _decimals(value)
    numbers = _numbers_in(text) if numbers is None else numbers
    factors = [1.0] + [factor for word, factor in _UNIT_SCALES.items() if word in unit.lower()]
    # Half-up, as people round; only a number with more digits than the value can be its origin.
    return any(
        n / f != value and int(n / f * scale + 0.5) / scale == value and n / f * scale != int(n / f * scale)
        for n in numbers
        for f in factors
    )


def verify_chart_sources(markdown: str, document_text: str, web_text: str = "") -> tuple[str, int]:
    """Label every chart by where its numbers are really found, whatever the model called it.

    "dokumen" when all values are written in the attached files, "web" when they are in the search
    results, otherwise "perkiraan". Returns the document and how many charts ended up as estimates.
    """
    estimates = 0
    texts = [("dokumen", document_text, _numbers_in(document_text)), ("web", web_text, _numbers_in(web_text))]

    def check(match: re.Match) -> str:
        nonlocal estimates
        body = match.group(1)
        raw = body.strip().removeprefix("```chart").removesuffix("```").strip()
        try:
            spec = json.loads(raw)
            values = [float(v) for v in spec["values"]]
            unit = str(spec.get("unit", ""))
        except (ValueError, TypeError, KeyError, AttributeError):
            return body  # malformed: the renderer drops it anyway
        origin = next(
            (name for name, text, numbers in texts if text and all(number_in_text(v, text, numbers, unit) for v in values)),
            "perkiraan",
        )
        estimates += origin == "perkiraan"
        spec["source"] = origin
        return "```chart\n" + json.dumps(spec, ensure_ascii=False) + "\n```"

    return _CHART_BLOCK.sub(check, markdown), estimates


def cite_sources(markdown: str, sources: list[WebSource], today: date) -> tuple[str, list[WebSource]]:
    """Renumber the [n] citations in reading order and append a reference list of the cited sources only.

    Any reference list the model wrote itself is discarded: titles and links come from the search
    results, never from the model's memory. Returns the document and the sources it cites.
    """
    markdown = _REFERENCES_HEADING.sub("", markdown).rstrip()
    order: list[int] = []

    def renumber(match: re.Match) -> str:
        numbers = [int(n) for n in re.split(r"\s*[,;]\s*", match.group(1))]
        if not all(1 <= n <= len(sources) for n in numbers):
            return match.group(0)  # not a citation of ours, e.g. a year in brackets
        for n in numbers:
            if n not in order:
                order.append(n)
        return "[" + ", ".join(str(order.index(n) + 1) for n in numbers) + "]"

    # Chart blocks hold JSON arrays such as [12, 8, 5]; those are not citations.
    parts = _CHART_BLOCK.split(markdown)
    markdown = "".join(part if part.startswith("```chart") else _CITATION.sub(renumber, part) for part in parts)

    cited = [sources[n - 1] for n in order]
    if cited:
        entries = "\n".join(
            f"{i}. {s.title.replace('|', '-')}. {s.site}. {s.url} (diakses {long_date(today)})." for i, s in enumerate(cited, start=1)
        )
        markdown += f"\n\n## Daftar Pustaka\n\n{entries}"
    return markdown, cited
