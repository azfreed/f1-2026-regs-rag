"""
Stage 1 of the FIA 2026 F1 Regulations RAG pipeline: PDF parsing.

For each section PDF, this script:
  1. Extracts text line-by-line per page (pdfplumber).
  2. Strips repeating headers/footers using a position-based band, cross-checked
     against known footer/header string patterns.
  3. Tags each body line with the nearest preceding article/clause anchor
     (e.g. B1.2.2), and records the dominant text color per line, since color
     marks amendment text in Section B's Appendix B5 and is needed for the
     stage 2 in-force-vs-future-amendment decision.

Verification behind these design choices (run against Section B exhaustively,
Section C on a sample): no duplicate text layer was found on any checked page
-- each page has exactly one text layer, not the two originally assumed.

A word-join glitch repair step (fixing things like "F1Car" -> "F1 Car") was
built, tested, and removed. A frequency-gated heuristic meant to avoid
splitting legitimate merged-case tokens (kWh, McLaren) instead corrupted
unit notation throughout Section C ("100kJ" -> "100k J", "TiAl6V4" ->
"Ti Al6V4") because short technical abbreviations and chemical symbols
picked up enough standalone frequency from lettered list enumerators and
element-symbol usage to pass the gate. Tightened enough to stop that, it
caught zero real errors -- the one confirmed instance in the corpus turned
out to already be correctly spaced by pdfplumber's line extraction, and the
other required loosening the gate back into corruption territory to catch.
Net effect: the risk consistently outweighed the benefit, so this step does
not run. Any word-join glitches in the output are unrepaired.

Output: one JSONL file per section in data/parsed/, one JSON object per page.
Also writes data/parsed/_stage1_log.json with pages flagged for manual review
(unconfirmed page codes, unrecognized header/footer text, pages with no
anchors found).

Usage:
    python -m src.stage1_parse
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import pdfplumber

# ---------------------------------------------------------------------------
# Corpus definition
# ---------------------------------------------------------------------------

PDF_DIR = Path("data/pdfs")
OUT_DIR = Path("data/parsed")

# section letter -> (filename, issue, date, title)
SECTIONS: dict[str, tuple[str, str, str, str]] = {
    "A": ("FIA 2026 F1 Regulations - Section A [General Provisions] - Iss 03 - 2026-06-25.pdf", "03", "2026-06-25", "General Provisions"),
    "B": ("FIA 2026 F1 Regulations - Section B [Sporting] - Iss 07 - 2026-06-25.pdf", "07", "2026-06-25", "Sporting"),
    "C": ("FIA 2026 F1 Regulations - Section C [Technical] - Iss 19 - 2026-06-25.pdf", "19", "2026-06-25", "Technical"),
    "D": ("FIA 2026 F1 Regulations - Section D [Financial - F1 Teams] - Iss 07 - 2026-06-25.pdf", "07", "2026-06-25", "Financial - F1 Teams"),
    "E": ("FIA 2026 F1 Regulations - Section E [Financial – PU Manufacturers] - Iss 06 - 2026-06-25.pdf", "06", "2026-06-25", "Financial - PU Manufacturers"),
    "F": ("FIA 2026 F1 Regulations - Section F [Operational] - Iss 09 - 2026-06-25.pdf", "09", "2026-06-25", "Operational"),
}

# ---------------------------------------------------------------------------
# Header / footer stripping
# ---------------------------------------------------------------------------
HEADER_BAND_PT = 40   # line top below this = header candidate

# The footer's absolute vertical position is NOT constant across sections --
# checked directly on page 5 of every section: A/B sit at top=792.7, C at
# 785.9, D at 786.3, F splits its title and code lines across 775.0 and
# 779.3. A single fixed distance-from-bottom threshold can't cleanly
# separate real footer from body across all six PDFs: the safe margin
# between the tightest confirmed body line (Section B, top=773.4) and the
# loosest real footer line (Section F, top=775.0) is under 2pt once every
# section is checked, not the ~19pt margin a single-section calibration
# suggested. An earlier version used a fixed 55pt band from Section B alone
# and silently leaked footer text into Section C's body on every one of its
# 257 pages as a result.
#
# The copyright line ("©2026 Fédération Internationale de l'Automobile") is
# an exact, unambiguous regex match and anchors the footer far more
# reliably: its position relative to the line(s) above it is a stable ~11pt
# gap across every section checked, even though its own absolute page
# position varies. Footer detection locates this line per page and treats
# anything within 20pt above it (generous buffer over the observed 11pt
# gap) as footer, rather than using a fixed page-position threshold.
FOOTER_ANCHOR_BUFFER_PT = 20

FOOTER_LINE_PATTERNS = [
    re.compile(r"©\s*20\d\d\s+Fédération Internationale de l.Automobile", re.IGNORECASE),
    re.compile(r"20\d\d Form\s?ula 1.*Regulations", re.IGNORECASE),
    re.compile(r"Issue \d+"),
    re.compile(r"\d{1,2} (January|February|March|April|May|June|July|August|September|October|November|December) 20\d\d"),
    re.compile(r"^[A-F]\s?\d{1,3}$"),  # page code, e.g. "B5", "B 5", "B34"
]
COPYRIGHT_PATTERN = FOOTER_LINE_PATTERNS[0]
HEADER_LINE_PATTERN = re.compile(r"^SECTION [A-F]:")

# Every page also carries a large decorative corner glyph (e.g. a 36pt "B"
# next to a 28pt "0") that is CONSTANT across all pages of a section -- it is
# not a page number and must not be mistaken for the real page code. The
# actual page code sits inside the footer, often merged onto the same text
# line as the section title and date (e.g. "...25 June 2026 D1 1"), with an
# extraction artifact inserting a stray space between its digits. Rather than
# pattern-guess which digits are the real code, the parser checks the footer
# band for the exact expected code -- section letter + the PDF's own page
# index -- since that's already known and confirmed to align 1:1 with the
# printed code on every section checked (A, B, D).

# ---------------------------------------------------------------------------
# Anchor detection
# ---------------------------------------------------------------------------
# Articles/clauses are lettered by section: B1, B1.2, B1.2.2. List items
# (a., b., c.) under a sub-clause are NOT given their own anchor here --
# they inherit the parent sub-clause's anchor. Revisit at stage 2 if
# citation granularity needs to go to the list-item level.


ANCHOR_PATTERN = re.compile(r"^([A-F]\d{1,2}(?:\.\d{1,3}){0,4})\b")


def find_anchor(line_text: str) -> str | None:
    m = ANCHOR_PATTERN.match(line_text.strip())
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Per-page parsing
# ---------------------------------------------------------------------------

@dataclass
class PageRecord:
    section: str
    pdf_page: int
    page_code: str | None
    lines: list[str]
    anchors: list[dict]       # [{"anchor": "B1.2.2", "line_index": 0}, ...]
    line_colors: list[list]   # dominant non_stroking_color per body line


def classify_line(line: dict, footer_zone_top: float) -> str:
    """Returns 'header', 'footer', or 'body' for a pdfplumber text line.
    footer_zone_top is the page's own copyright-line top minus a buffer
    (see FOOTER_ANCHOR_BUFFER_PT), or a page-height-based fallback if no
    copyright line was found on this page."""
    if line["top"] < HEADER_BAND_PT:
        return "header"
    if line["top"] >= footer_zone_top:
        return "footer"
    return "body"


def dominant_color(page: "pdfplumber.page.Page", line: dict) -> list:
    """Most common non_stroking_color among chars in this line's bbox,
    rounded to 3 decimals. Needed at stage 2 to separate in-force text
    from colored amendment/strikethrough text in Appendix B5."""
    chars = page.crop((line["x0"], line["top"], line["x1"], line["bottom"]), relative=False).chars
    colors = Counter()
    for c in chars:
        col = c.get("non_stroking_color")
        if col:
            colors[tuple(round(x, 3) for x in col)] += 1
    if not colors:
        return [0, 0, 0]
    return list(colors.most_common(1)[0][0])


def parse_pdf(path: Path, section: str, log: list) -> list[PageRecord]:
    records: list[PageRecord] = []

    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            lines = page.extract_text_lines()

            copyright_top = None
            for line in lines:
                if COPYRIGHT_PATTERN.search(line["text"]):
                    copyright_top = line["top"]
                    break

            if copyright_top is not None:
                footer_zone_top = copyright_top - FOOTER_ANCHOR_BUFFER_PT
            else:
                # No copyright line found on this page (cover/TOC pages, or
                # a layout that doesn't match) -- fall back to a
                # page-height-based estimate and flag it, since the dynamic
                # anchor couldn't be computed here.
                footer_zone_top = page.height - 55
                log.append({
                    "action": "footer_anchor_not_found",
                    "section": section, "page": page_no,
                })

            page_code = None
            expected_code = f"{section}{page_no}"
            body_lines: list[str] = []
            body_colors: list[list] = []
            anchors: list[dict] = []

            for line in lines:
                cls = classify_line(line, footer_zone_top)

                if cls in ("header", "footer"):
                    is_confirmed_code = False
                    if cls == "footer":
                        stripped = re.sub(r"\s+", "", line["text"])
                        if expected_code in stripped:
                            page_code = expected_code
                            is_confirmed_code = True
                    # Flag anything caught in the band that doesn't match a
                    # known header/footer pattern -- could be misclassified
                    # body text on an unusual page layout. The constant
                    # decorative corner glyph (e.g. lone "B", lone "0") and a
                    # confirmed page code (whatever split form it takes) are
                    # expected here and excluded from this check.
                    known = (
                        HEADER_LINE_PATTERN.search(line["text"])
                        or any(p.search(line["text"]) for p in FOOTER_LINE_PATTERNS)
                        or re.fullmatch(r"[A-F]|\d", line["text"].strip())
                        or is_confirmed_code
                    )
                    if not known and len(line["text"].strip()) > 3:
                        log.append({
                            "action": "unrecognized_header_footer_band_text",
                            "section": section, "page": page_no,
                            "text": line["text"], "classified_as": cls,
                        })
                    continue

                anchor = find_anchor(line["text"])
                if anchor:
                    anchors.append({"anchor": anchor, "line_index": len(body_lines)})
                body_lines.append(line["text"])
                body_colors.append(dominant_color(page, line))

            if page_code is None:
                log.append({
                    "action": "page_code_not_confirmed",
                    "section": section, "page": page_no, "expected": expected_code,
                })

            if not anchors and page_no > 3:
                log.append({
                    "action": "no_anchors_on_page", "section": section, "page": page_no,
                })

            records.append(PageRecord(
                section=section, pdf_page=page_no, page_code=page_code,
                lines=body_lines, anchors=anchors, line_colors=body_colors,
            ))

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full_log: list = []
    summary = []

    for letter, (filename, issue, date, title) in SECTIONS.items():
        path = PDF_DIR / filename
        print(f"Parsing Section {letter} ({title}), Issue {issue}, {date} ...")
        records = parse_pdf(path, letter, full_log)

        out_path = OUT_DIR / f"section_{letter}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

        anchor_count = sum(len(r.anchors) for r in records)
        no_anchor_pages = sum(1 for r in records if not r.anchors and r.pdf_page > 3)
        summary.append({
            "section": letter, "title": title, "issue": issue, "date": date,
            "pages": len(records), "anchors_found": anchor_count,
            "pages_without_anchors": no_anchor_pages,
        })
        print(f"  {len(records)} pages, {anchor_count} anchors, "
              f"{no_anchor_pages} pages with no anchor found -> {out_path}")

    with (OUT_DIR / "_stage1_log.json").open("w", encoding="utf-8") as f:
        json.dump(full_log, f, ensure_ascii=False, indent=2)

    with (OUT_DIR / "_stage1_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    unrecognized = sum(1 for e in full_log if e.get("action") == "unrecognized_header_footer_band_text")
    unconfirmed_codes = sum(1 for e in full_log if e.get("action") == "page_code_not_confirmed")
    no_anchor_pages = sum(1 for e in full_log if e.get("action") == "no_anchors_on_page")
    print(f"\nUnrecognized text caught in header/footer bands: {unrecognized}")
    print(f"Pages with unconfirmed page code: {unconfirmed_codes}")
    print(f"Pages with no anchors found (page > 3): {no_anchor_pages}")
    print("Review data/parsed/_stage1_log.json for details.")


if __name__ == "__main__":
    main()
