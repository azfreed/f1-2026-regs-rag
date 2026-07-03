# f1-2026-regs-rag

RAG (Retrieval-Augmented Generation) pipeline over the FIA 2026 Formula 1 Regulations.

## Corpus

All six regulation sections, extracted from PDFs uploaded to `data/pdfs/`.

| Section | Title | Issue | Date |
|---|---|---|---|
| A | General Provisions | 03 | 2026-06-25 |
| B | Sporting | 07 | 2026-06-25 |
| C | Technical | 19 | 2026-06-25 |
| D | Financial – F1 Teams | 07 | 2026-06-25 |
| E | Financial – PU Manufacturers | 06 | 2026-06-25 |
| F | Operational | 09 | 2026-06-25 |

Source: FIA regulation PDFs. The build environment cannot reach fia.com directly, so PDFs are pulled from this repo (`data/pdfs/`) rather than fetched at build time.

## Models

- **GTE-small** (General Text Embeddings): primary embedding model, stages 1–4.
- **BGE-M3** (BAAI General Embeddings): comparison model, later stage.

## Pipeline

1. **Parse** — extract text per page, strip headers/footers, tag article/clause anchors.
2. **Chunk** — split into retrieval units with article-number metadata.
3. **Index** — GTE-small embeddings to Chroma, plus parallel BM25.
4. **Retrieve** — dense / sparse / hybrid via Reciprocal Rank Fusion (RRF).
5. **Generate** — grounded generation with article citations.
6. **Eval** (optional) — recall@k, faithfulness.

## Status

### Stage 1 — Parse: done

`src/stage1_parse.py`. Output: `data/parsed/section_{A-F}.jsonl`, one JSON record per page (section, PDF page number, confirmed page code, body lines, article/clause anchors, per-line dominant text color).

Verified across all 596 pages, all six sections:

| Section | Pages | Anchors found | Pages with no anchor |
|---|---|---|---|
| A | 84 | 329 | 30 |
| B | 99 | 485 | 13 |
| C | 257 | 1,158 | 70 |
| D | 64 | 301 | 22 |
| E | 60 | 191 | 26 |
| F | 32 | 112 | 4 |

"Pages with no anchor" were spot-checked, not assumed — they're definitions appendices, list continuations, void articles, and future-year-change appendices, none of which carry numbered clause anchors on that particular page.

Automated checks, corpus-wide: 0 unconfirmed page codes, 0 unrecognized header/footer text, 0 pages where the footer-anchor (copyright line) couldn't be found.

**Design notes:**
- Header/footer stripping is anchored to the copyright line's position, not a fixed page coordinate. The footer's absolute position varies by section (checked directly: A/B at one position, C and D each a few points off, F split across two more). A single fixed threshold, calibrated from one section, was found to silently leak footer text into 257 pages of Section C body content before this fix.
- Page codes (e.g. `B34`) are confirmed against the PDF's own page index rather than pattern-matched — a constant decorative corner glyph on every page was initially mistaken for the real code.
- A word-join glitch repair step (e.g. `F1Car` → `F1 Car`) was built, tested, and removed. A frequency-gated heuristic meant to avoid corrupting merged-case tokens (`kWh`, `McLaren`) instead corrupted unit notation and alloy codes throughout Section C (`100kJ` → `100k J`, `TiAl6V4` → `Ti Al6V4`). Tightened enough to stop that, it caught zero real errors. Output text is unrepaired; any word-join glitches in the source PDFs pass through as-is.
- Per-line text color is captured but unused until stage 2 — needed for the in-force-vs-future-amendment decision below.

### Future-year amendments (Appendix B5 / A9 / F2, and possibly others)

At least three sections carry an appendix of pre-approved changes for future years (2027–2029) that describe rules not yet in force: Section B's Appendix B5, Section A's Appendix A9, Section F's Appendix F2. These describe rules that will supersede the 2026 text — indexing them undifferentiated risks retrieval surfacing a future rule as if it were current.

Decision: tag rather than exclude. Each chunk gets a `status` field (`in_force` / `future_amendment`) determined at stage 2 from the per-line color data already captured in stage 1. Default retrieval filters to `in_force`; a query specifically about future changes can opt in to `future_amendment`. This preserves the ability to answer "what changes in 2027?" without a reparse, and keeps the default answer grounded in the rules currently in effect.

## Environment

- PDFs are supplied via this repo, not fetched at runtime — the sandbox cannot reach fia.com.
- `pip install -r requirements.txt`
- `python -m src.stage1_parse`

## Repo structure

```
data/
  pdfs/         source PDFs (6 sections)
  parsed/       stage 1 output: section_{A-F}.jsonl, _stage1_log.json, _stage1_summary.json
src/
  stage1_parse.py
notebooks/
  rag_test.py   early scratch notebook, not part of the pipeline
```
