"""Phase 2: assign the text of every body page to its sthana and lesson.

Input : data/processed/pages.jsonl         (Phase 1)
Output: data/processed/lesson_pages.jsonl  (one record per lesson per page)
        data/processed/structure.json      (lessons with page spans, for review)

How it works

1. Body vs. wrapper pages. The translation came out in monthly parts, and each
   part's wrapper (title page, notices, press opinions, adverts) is bound in
   with the text. The transcriber typed many wrapper pages as "text". Body
   pages carry the running header "CHARAKA-SAMHITA" (or a sthana name);
   wrapper pages do not. A page that opens a lesson often has no header, so a
   lesson heading near the top also makes a page a body page.
2. Lesson starts. A line "LESSON <roman>" starts the next expected lesson.
   Numbering restarts at I in every sthana, so "LESSON I" moves on to the next
   sthana in the order this translation printed them. A heading that is not
   the next expected lesson is ignored and reported. LESSON_START_FIXES lists
   the few lesson starts the transcription leaves unmarked or misnumbered.
3. Validation. Every lesson in STHANAS must be found once, in order.

Run:  python -m src.structure
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src import config
from src.ingest import PageRecord, is_indexable, load_pages


@dataclass(frozen=True)
class Sthana:
    key: str                  # short id used in records, e.g. "sutra"
    name: str                 # display name, e.g. "Sutra Sthana"
    subject: str              # what the division is about, in plain English
    lessons: tuple[str, ...]  # standard Sanskrit lesson titles, in order


# The divisions in the order this translation printed them (the standard order
# is Sutra, Nidana, Vimana, Sharira, Indriya, Chikitsa, Kalpa, Siddhi). Kalpa,
# Siddhi and Chikitsa lessons XXIII-XXX were never translated.
STHANAS: tuple[Sthana, ...] = (
    Sthana("sutra", "Sutra Sthana", "general principles", (
        "Dirghanjivitiya", "Apamarga-Tanduliya", "Aragvadhiya",
        "Shadvirechanashatashritiya", "Matrashitiya", "Tasyashitiya",
        "Navegandharaniya", "Indriyopakramaniya", "Khuddakachatushpada",
        "Mahachatushpada", "Tistraishaniya", "Vatakalakaliya", "Sneha",
        "Sveda", "Upakalpaniya", "Chikitsaprabhritiya", "Kiyantahshirasiya",
        "Trishothiya", "Ashtodariya", "Maharoga", "Ashtauninditiya",
        "Langhana-Brimhaniya", "Santarpaniya", "Vidhishonitiya",
        "Yajjahpurushiya", "Atreyabhadrakapyiya", "Annapanavidhi",
        "Vividhashitapitiya", "Dashapranayataniya", "Arthedashamahamuliya",
    )),
    Sthana("vimana", "Vimana Sthana", "specific measures and assessment", (
        "Rasa", "Trividhakukshiya", "Janapadodhvamsaniya",
        "Trividharogavisheshavijnaniya", "Srotas", "Roganika",
        "Vyadhitarupiya", "Rogabhishagjitiya",
    )),
    Sthana("sharira", "Sharira Sthana", "the body", (
        "Katidhapurushiya", "Atulyagotriya", "Khuddikagarbhavakranti",
        "Mahatigarbhavakranti", "Purushavichaya", "Shariravichaya",
        "Sharirasankhya", "Jatisutriya",
    )),
    Sthana("indriya", "Indriya Sthana", "signs of prognosis", (
        "Varnasvariya", "Pushpitaka", "Parimarshaniya", "Indriyanika",
        "Purvarupiya", "Katamanishariya", "Pannarupiya", "Avakshirasiya",
        "Yasyashyavanimittiya", "Sadyomaraniya", "Anujyotiya", "Gomayachurniya",
    )),
    Sthana("nidana", "Nidana Sthana", "causes and diagnosis of disease", (
        "Jvara Nidana", "Raktapitta Nidana", "Gulma Nidana", "Prameha Nidana",
        "Kushtha Nidana", "Shosha Nidana", "Unmada Nidana", "Apasmara Nidana",
    )),
    Sthana("chikitsa", "Chikitsa Sthana", "treatment", (
        "Rasayana", "Vajikarana", "Jvara Chikitsa", "Raktapitta Chikitsa",
        "Gulma Chikitsa", "Prameha Chikitsa", "Kushtha Chikitsa",
        "Rajayakshma Chikitsa", "Unmada Chikitsa", "Apasmara Chikitsa",
        "Kshatakshina Chikitsa", "Shvayathu Chikitsa", "Udara Chikitsa",
        "Arsha Chikitsa", "Grahani Chikitsa", "Pandu Chikitsa",
        "Hikka-Shvasa Chikitsa", "Kasa Chikitsa", "Atisara Chikitsa",
        "Chhardi Chikitsa", "Visarpa Chikitsa", "Trishna Chikitsa",
    )),
)

# (sthana, lesson) -> (scan page, exact line where the lesson starts), for
# the verified copy of the source (config.SOURCE_CONTENT_SHA256).
LESSON_START_FIXES: dict[tuple[str, int], tuple[int, str]] = {
    ("sutra", 15): (246, "SECTION XV."),             # printed "SECTION", not "LESSON"
    ("sutra", 16): (260, "SECTION XVI."),
    ("sutra", 19): (310, "LESSON IXX."),             # misprint for XIX
    ("indriya", 1): (1107, "I N D R I Y A S T H A N A M."),  # no lesson heading
}

LESSON_HEADING_RE = re.compile(r"^LESSON\s+([IVXL]+)\s*[.:]?$")
# Loose on purpose: headers are OCR'd as "CAARAKA-SAMHITA", "CHARAK SAMHITA", ...
BODY_HEADER_RE = re.compile(r"CHA?RAK|SAM\w*H|STH[AĀ]N", re.IGNORECASE)
OPENING_LINES = 4        # a lesson heading this near the top makes a page a body page
MAX_HEADING_LINES = 3    # title lines above "LESSON N" that belong to the new lesson


def to_roman(n: int) -> str:
    tens = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"]
    ones = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
    return tens[n // 10] + ones[n % 10]


ROMAN_VALUES = {to_roman(n): n for n in range(1, 100)}  # strict: "IXX" is not here


def lesson_label(sthana_key: str, lesson: int, sthanas: tuple[Sthana, ...] = STHANAS) -> str:
    """Human-readable lesson name, e.g. "Sutra Sthana, Lesson I (Dirghanjivitiya)"."""
    sthana = next(s for s in sthanas if s.key == sthana_key)
    return f"{sthana.name}, Lesson {to_roman(lesson)} ({sthana.lessons[lesson - 1]})"


@dataclass
class LessonPage:
    """The part of one scan page that belongs to one lesson."""
    source_id: str
    sthana: str                  # Sthana.key
    lesson: int                  # 1-based, within the sthana
    lesson_title: str
    scan_page: int
    printed_page: str | None
    main_text: str
    notes_text: str              # the page's notes, on its longest piece only
    flags: list[str]             # copied from the page record


@dataclass
class LessonStart:
    sthana: str
    lesson: int
    scan_page: int
    offset: int                  # character offset in the page's main_text


@dataclass
class Structure:
    starts: list[LessonStart]
    lesson_pages: list[LessonPage]
    missing: list[tuple[str, int]]       # expected lessons that were never found
    ignored_headings: list[str]          # headings out of sequence
    wrapper_pages: list[int] = field(default_factory=list)  # text pages left out


def _lines_with_offsets(text: str) -> list[tuple[int, str]]:
    result, offset = [], 0
    for line in text.split("\n"):
        result.append((offset, line))
        offset += len(line) + 1
    return result


def _is_title_line(line: str) -> bool:
    return any(c.isalpha() for c in line) and line == line.upper() and len(line) <= 60


def _start_offset(lines: list[tuple[int, str]], index: int) -> int:
    """Move a lesson start up over title lines such as "THE PLACE OF APHORISMS."."""
    taken = 0
    while index > 0 and taken < MAX_HEADING_LINES:
        prev = index - 1
        while prev >= 0 and not lines[prev][1].strip():
            prev -= 1
        if prev < 0 or not _is_title_line(lines[prev][1]):
            break
        index, taken = prev, taken + 1
    return lines[index][0]


def is_body_page(page: PageRecord, fix_pages: set[int] = frozenset()) -> bool:
    if not is_indexable(page):
        return False
    if page.header and BODY_HEADER_RE.search(page.header):
        return True
    opening = [l for l in page.main_text.split("\n") if l.strip()][:OPENING_LINES]
    return page.scan_page in fix_pages or any(LESSON_HEADING_RE.match(l) for l in opening)


def build_structure(
    pages: list[PageRecord],
    sthanas: tuple[Sthana, ...] = STHANAS,
    fixes: dict[tuple[str, int], tuple[int, str]] = LESSON_START_FIXES,
) -> Structure:
    expected = [(s.key, n) for s in sthanas for n in range(1, len(s.lessons) + 1)]
    titles = {(s.key, n): t for s in sthanas for n, t in enumerate(s.lessons, 1)}
    fixes_by_page: dict[int, dict[str, tuple[str, int]]] = {}
    for key, (scan_page, line) in fixes.items():
        fixes_by_page.setdefault(scan_page, {})[line] = key

    pages = sorted(pages, key=lambda p: p.scan_page)
    body = [p for p in pages if is_body_page(p, set(fixes_by_page))]
    body_numbers = {p.scan_page for p in body}
    wrappers = [p.scan_page for p in pages if is_indexable(p) and p.scan_page not in body_numbers]

    starts: list[LessonStart] = []
    ignored: list[str] = []
    position = 0  # index into `expected` of the next lesson to find
    for page in body:
        page_fixes = dict(fixes_by_page.get(page.scan_page, {}))
        lines = _lines_with_offsets(page.main_text)
        for index, (_, line) in enumerate(lines):
            line = line.strip()
            nxt = expected[position] if position < len(expected) else None
            if line in page_fixes:
                key = page_fixes.pop(line)
                if key != nxt:
                    raise ValueError(f"fix {key} on page {page.scan_page} is out of order; "
                                     f"expected {nxt}")
            elif match := LESSON_HEADING_RE.match(line):
                number = ROMAN_VALUES.get(match.group(1))
                if nxt is None or number != nxt[1]:
                    ignored.append(f"page {page.scan_page}: '{line}' (expected "
                                   f"{'nothing' if nxt is None else lesson_label(*nxt, sthanas)})")
                    continue
                key = nxt
            else:
                continue
            starts.append(LessonStart(key[0], key[1], page.scan_page, _start_offset(lines, index)))
            position += 1
        if page_fixes:
            raise ValueError(f"fix line(s) not found on page {page.scan_page}: {list(page_fixes)}")

    lesson_pages = _split_pages(body, starts, titles)
    return Structure(starts, lesson_pages, expected[position:], ignored, wrappers)


def _split_pages(body: list[PageRecord], starts: list[LessonStart],
                 titles: dict[tuple[str, int], str]) -> list[LessonPage]:
    """Cut each body page at the lesson starts on it; drop text before the first lesson."""
    starts_by_page: dict[int, list[LessonStart]] = {}
    for start in starts:
        starts_by_page.setdefault(start.scan_page, []).append(start)

    records: list[LessonPage] = []
    current: tuple[str, int] | None = None
    for page in body:
        cuts = [(0, current)] + [(s.offset, (s.sthana, s.lesson)) for s in starts_by_page.get(page.scan_page, [])]
        pieces = []
        for (begin, key), (end, _) in zip(cuts, cuts[1:] + [(len(page.main_text), None)]):
            text = page.main_text[begin:end].strip()
            if key is not None and text:
                pieces.append(LessonPage(
                    source_id=page.source_id, sthana=key[0], lesson=key[1],
                    lesson_title=titles[key], scan_page=page.scan_page,
                    printed_page=page.printed_page, main_text=text, notes_text="",
                    flags=list(page.flags),
                ))
        if pieces:
            max(pieces, key=lambda p: len(p.main_text)).notes_text = page.notes_text
        records.extend(pieces)
        current = cuts[-1][1]
    return records


def lesson_spans(structure: Structure) -> list[dict]:
    """One summary per lesson: first and last scan page, pages, characters."""
    spans: dict[tuple[str, int], dict] = {}
    for record in structure.lesson_pages:
        span = spans.setdefault((record.sthana, record.lesson), {
            "sthana": record.sthana, "lesson": record.lesson, "title": record.lesson_title,
            "first_page": record.scan_page, "last_page": record.scan_page,
            "pages": 0, "chars": 0,
        })
        span["last_page"] = record.scan_page
        span["pages"] += 1
        span["chars"] += len(record.main_text)
    return list(spans.values())


def write_outputs(structure: Structure, lesson_pages_path: Path, structure_path: Path) -> None:
    lesson_pages_path.parent.mkdir(parents=True, exist_ok=True)
    with lesson_pages_path.open("w", encoding="utf-8") as f:
        for record in structure.lesson_pages:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    summary = {
        "source_id": config.SOURCE_ID,
        "sthanas": [{"key": s.key, "name": s.name, "subject": s.subject} for s in STHANAS],
        "lessons": lesson_spans(structure),
        "missing_lessons": [lesson_label(*key) for key in structure.missing],
        "ignored_headings": structure.ignored_headings,
        "wrapper_pages": structure.wrapper_pages,
    }
    structure_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_lesson_pages(path: Path = config.LESSON_PAGES_PATH) -> list[LessonPage]:
    with path.open(encoding="utf-8") as f:
        return [LessonPage(**json.loads(line)) for line in f if line.strip()]


def print_report(structure: Structure) -> None:
    spans = lesson_spans(structure)
    for sthana in STHANAS:
        own = [s for s in spans if s["sthana"] == sthana.key]
        if own:
            print(f"{sthana.name:<16}: lessons {len(own)}/{len(sthana.lessons)}, "
                  f"scan pages {own[0]['first_page']}-{own[-1]['last_page']}, "
                  f"{sum(s['chars'] for s in own):,} chars")
        else:
            print(f"{sthana.name:<16}: not found")
    split = len(structure.lesson_pages) - len({r.scan_page for r in structure.lesson_pages})
    print(f"Lesson-page records: {len(structure.lesson_pages)} ({split} pages shared by two lessons)")
    wrappers = structure.wrapper_pages
    more = f" ... (+{len(wrappers) - 15})" if len(wrappers) > 15 else ""
    print(f"Wrapper pages left out: {len(wrappers)} {wrappers[:15]}{more}")
    for line in structure.ignored_headings:
        print(f"  ignored heading {line}")
    for key in structure.missing:
        print(f"  MISSING {lesson_label(*key)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assign page text to sthanas and lessons.")
    parser.add_argument("--pages", type=Path, default=config.PAGES_PATH)
    parser.add_argument("--out", type=Path, default=config.LESSON_PAGES_PATH)
    parser.add_argument("--structure", type=Path, default=config.STRUCTURE_PATH)
    args = parser.parse_args(argv)

    if not args.pages.exists():
        print(f"Page records not found: {args.pages}")
        print("Run first: python -m src.ingest")
        return 1

    structure = build_structure(load_pages(args.pages))
    print_report(structure)
    if structure.missing:
        print("\nNot all lessons were found; nothing written. "
              "Check the headings above and LESSON_START_FIXES.")
        return 1
    write_outputs(structure, args.out, args.structure)
    print(f"\nWrote {len(structure.lesson_pages)} records to {args.out}")
    print(f"Wrote lesson summary to {args.structure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
