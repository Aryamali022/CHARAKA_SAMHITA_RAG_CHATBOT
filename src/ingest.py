"""Phase 1: parse the Source Library transcription into clean page records.

Input : data/raw/charaka-samhita-ocr.txt  ("[Page N]" markers + XML-like tags)
Output: data/processed/pages.jsonl        (one JSON object per scan page)

This module only parses and cleans text. It does not decide which sthana or
lesson a page belongs to (Phase 2) and does not chunk (Phase 3).

Run:  python -m src.ingest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from src import config

PAGE_MARKER_RE = re.compile(r"^\[Page (\d+)\][ \t]*$", re.MULTILINE)
# "────" closes every page; "════" opens the document footer after the last page.
PAGE_RULE_RE = re.compile(r"^[─═]+[ \t]*$", re.MULTILINE)
FOOTER_RULE_RE = re.compile(r"^═+[ \t]*$", re.MULTILINE)

# Page-level metadata tags: their values are read, then they are removed.
METADATA_TAGS = (
    "scan-quality", "language", "script", "page-type", "page-num",
    "header", "warning", "meta", "vocab", "image-desc", "sig",
)
SIDE_NOTE_RE = re.compile(r"<(note|margin)\b[^>]*>(.*?)</\1>", re.DOTALL)
MISSING_VALUES = {"", "n/a", "none", "null"}

# A footnote line starts with a reference mark: "* text", "\* text", "† text",
# or a lettered note like "*a.* text" / "*(a).* text". A line merely written
# in italics ("*(Here are some verses).*") does not count.
FOOTNOTE_START_RE = re.compile(r"^\\?(\*\s|[†‡§¶]|\*\(?[a-z]\)?\.?\*?\s)")
# Lines that follow a decorative "---" rule rather than a footnote rule.
HEADING_LIKE_RE = re.compile(
    r"^(#|->|\||\*\*[^a-z*]+\*\*\s*$|\**\s*(LESSON|SECTION|CHAPTER|THE PLACE|THE DIVISION)\b)"
)

MIN_TEXT_CHARS = 50          # a text page shorter than this is flagged "empty"
MIN_LETTER_RATIO = 0.6       # below this share of letters a page is flagged "garbled"


@dataclass
class PageRecord:
    source_id: str
    scan_page: int               # position in the scan, 1..N, unique
    printed_page: str | None     # number printed on the page; restarts per part
    page_type: str               # text, preface, title-page, blank, toc, ...
    scan_quality: str | None
    language: str | None
    header: str | None           # running header, e.g. "CHARAKA-SAMHITA."
    main_text: str
    notes_text: str              # translator's footnotes and side notes
    vocab: list[str]             # key terms listed by the transcriber
    unclear_count: int           # words the transcriber marked <unclear>
    flags: list[str]


def content_sha256(raw: bytes) -> str:
    """Checksum of the book content, ignoring the download header.

    The header contains the download date, so two downloads of the same book
    differ byte for byte. Hashing from the first "[Page 1]" marker, with line
    endings normalised, gives a fingerprint that is stable across downloads
    and across Windows/Unix line endings.
    """
    raw = raw.replace(b"\r\n", b"\n")
    start = raw.find(b"[Page 1]")
    if start == -1:
        raise ValueError("no '[Page 1]' marker found; is this the Source Library text?")
    return hashlib.sha256(raw[start:]).hexdigest()


def split_pages(text: str) -> list[tuple[int, str]]:
    """Split the whole file into (scan_page, body) pairs.

    Text before the first marker is the download header and text after the
    "════" rule on the last page is the footer; both are discarded.
    """
    markers = list(PAGE_MARKER_RE.finditer(text))
    if not markers:
        raise ValueError("no '[Page N]' markers found")

    pages: list[tuple[int, str]] = []
    seen: set[int] = set()
    for i, marker in enumerate(markers):
        number = int(marker.group(1))
        if number in seen:
            raise ValueError(f"duplicate page marker: [Page {number}]")
        seen.add(number)

        is_last = i == len(markers) - 1
        end = len(text) if is_last else markers[i + 1].start()
        body = text[marker.end():end]
        if is_last:
            footer = FOOTER_RULE_RE.search(body)
            if footer:
                body = body[:footer.start()]
        pages.append((number, PAGE_RULE_RE.sub("", body).strip()))
    return pages


def split_main_and_notes(body: str) -> tuple[str, str, str]:
    """Split a page into (main, notes, method).

    The transcription usually starts the footnote area with a "---" line, but
    "---" is also used as a decorative rule before headings. We split at the
    first "---" whose next non-empty line does not look like a heading
    (method "rule"). Without such a rule we fall back to the first line that
    starts with a footnote mark (method "marker"); footnote text continued
    from the previous page has no mark, so it may stay in the main text.
    Otherwise the whole page is main text (method "none").
    """
    lines = body.split("\n")
    for i, line in enumerate(lines):
        if line.strip() != "---":
            continue
        following = next((l.strip() for l in lines[i + 1:] if l.strip()), "")
        if following and not HEADING_LIKE_RE.match(following):
            return "\n".join(lines[:i]), "\n".join(lines[i + 1:]), "rule"
    for i, line in enumerate(lines):
        if FOOTNOTE_START_RE.match(line.strip()):
            return "\n".join(lines[:i]), "\n".join(lines[i:]), "marker"
    return body, "", "none"


def clean_text(text: str) -> str:
    """Turn transcription markup into plain text for search and display."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"<sup\b[^>]*>.*?</sup>", "", text, flags=re.DOTALL)  # reference marks
    text = re.sub(r"</?[A-Za-z][\w-]*(\s[^>]*)?>", "", text)  # other tags: keep their content
    text = re.sub(r"\\([*_\[\]#|])", r"\1", text)  # markdown escapes like \*

    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if re.fullmatch(r"-{3,}|[─═]+", line) or re.fullmatch(r"\|?[\s:|-]*\|[\s:|-]*", line):
            continue  # horizontal rules and empty/separator table rows
        line = re.sub(r"^#+\s*", "", line)  # markdown heading marks
        line = re.sub(r"^->\s*|\s*<-$", "", line)  # ->centred<- lines
        line = line.replace("|", " ")  # table cells
        line = re.sub(r"[*†‡§¶⁰¹²³⁴⁵⁶⁷⁸⁹]", "", line)  # emphasis and footnote marks
        line = re.sub(r"\(\s*\)", "", line)  # brackets left empty by removed marks
        lines.append(re.sub(r"[ \t]+", " ", line).strip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _tag_value(body: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", body, re.DOTALL)
    if not match or match.group(1).strip().lower() in MISSING_VALUES:
        return None
    return match.group(1).strip()


def _letter_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    return sum(c.isalpha() for c in chars) / len(chars) if chars else 0.0


def parse_page(scan_page: int, body: str, source_id: str = config.SOURCE_ID) -> PageRecord:
    """Turn one raw page body into a PageRecord."""
    page_type = _tag_value(body, "page-type") or "unknown"
    vocab_raw = _tag_value(body, "vocab") or ""
    record_meta = {
        "printed_page": _tag_value(body, "page-num"),
        "scan_quality": _tag_value(body, "scan-quality"),
        "language": _tag_value(body, "language"),
        "header": _tag_value(body, "header"),
    }
    unclear_count = len(re.findall(r"<unclear\b", body))

    tags = "|".join(re.escape(t) for t in METADATA_TAGS)
    body = re.sub(rf"<({tags})\b[^>]*>.*?</\1>", "", body, flags=re.DOTALL)

    side_notes = []
    def _collect(match: re.Match) -> str:
        content = match.group(2).strip()
        if content and not content.isdigit() and "page is blank" not in content.lower():
            side_notes.append(content)
        return ""
    body = SIDE_NOTE_RE.sub(_collect, body)

    raw_main, raw_notes, split_method = split_main_and_notes(body)
    main_text = clean_text(raw_main)
    notes_text = clean_text("\n\n".join([raw_notes, *side_notes]))

    flags = []
    is_text = page_type == "text"
    if is_text and len(main_text) < MIN_TEXT_CHARS:
        flags.append("empty")
    if is_text and main_text and _letter_ratio(main_text) < MIN_LETTER_RATIO:
        flags.append("garbled")
    if record_meta["scan_quality"] in ("poor", "fair"):
        flags.append("low_scan_quality")
    if unclear_count:
        flags.append("has_unclear")
    if is_text and split_method == "marker":
        flags.append("notes_split_by_marker")
    if is_text and any(FOOTNOTE_START_RE.match(l.strip()) for l in raw_main.split("\n")):
        flags.append("notes_not_separated")
    if is_text and record_meta["printed_page"] is None:
        flags.append("missing_printed_page")

    return PageRecord(
        source_id=source_id,
        scan_page=scan_page,
        page_type=page_type,
        main_text=main_text,
        notes_text=notes_text,
        vocab=[v.strip() for v in vocab_raw.split(",") if v.strip()],
        unclear_count=unclear_count,
        flags=flags,
        **record_meta,
    )


def is_indexable(record: PageRecord) -> bool:
    """Candidate for the search index: a real text page with some content."""
    return record.page_type == "text" and "empty" not in record.flags


def ingest_file(path: Path) -> list[PageRecord]:
    text = path.read_text(encoding="utf-8")
    return [parse_page(number, body) for number, body in split_pages(text)]


def write_jsonl(records: list[PageRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def load_pages(path: Path = config.PAGES_PATH) -> list[PageRecord]:
    with path.open(encoding="utf-8") as f:
        return [PageRecord(**json.loads(line)) for line in f if line.strip()]


def print_report(records: list[PageRecord]) -> None:
    numbers = [r.scan_page for r in records]
    missing = sorted(set(range(min(numbers), max(numbers) + 1)) - set(numbers))
    text_pages = [r for r in records if r.page_type == "text"]
    types = Counter(r.page_type for r in records)
    flags = Counter(f for r in records for f in r.flags)

    print(f"Pages parsed        : {len(records)} (scan pages {min(numbers)}-{max(numbers)}, "
          f"{len(missing)} missing)")
    print("Page types          : " + ", ".join(f"{t} {n}" for t, n in types.most_common()))
    print(f"Indexable pages     : {sum(is_indexable(r) for r in records)}")
    print(f"Footnotes separated : {sum(bool(r.notes_text) for r in text_pages)} "
          f"of {len(text_pages)} text pages")
    print(f"Characters          : main {sum(len(r.main_text) for r in text_pages):,} / "
          f"notes {sum(len(r.notes_text) for r in text_pages):,} (text pages)")
    print("Flags               : " + (", ".join(f"{f} {n}" for f, n in flags.most_common()) or "none"))
    for flag in flags:
        pages = [r.scan_page for r in records if flag in r.flags]
        more = f" ... (+{len(pages) - 15})" if len(pages) > 15 else ""
        print(f"  {flag:<22}: {pages[:15]}{more}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse the source text into page records.")
    parser.add_argument("--source", type=Path, default=config.SOURCE_PATH)
    parser.add_argument("--out", type=Path, default=config.PAGES_PATH)
    args = parser.parse_args(argv)

    if not args.source.exists():
        print(f"Source file not found: {args.source}")
        print("See README (Source), then run: python -m scripts.verify_source")
        return 1
    if content_sha256(args.source.read_bytes()) != config.SOURCE_CONTENT_SHA256:
        print("WARNING: source differs from the verified copy (run: python -m scripts.verify_source)\n")

    records = ingest_file(args.source)
    write_jsonl(records, args.out)
    print_report(records)
    print(f"\nWrote {len(records)} page records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
