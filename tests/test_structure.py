import pytest

from src import config
from src.ingest import PageRecord, ingest_file
from src.structure import (
    LESSON_START_FIXES,
    ROMAN_VALUES,
    STHANAS,
    Sthana,
    build_structure,
    lesson_label,
    load_lesson_pages,
    to_roman,
    write_outputs,
)

TINY = (
    Sthana("a", "A Sthana", "first things", ("A-one", "A-two")),
    Sthana("b", "B Sthana", "second things", ("B-one",)),
)
TINY_FIXES = {("b", 1): (6, "B STHANAM.")}


def page(n, text, header="CHARAKA-SAMHITA.", notes=""):
    return PageRecord(
        source_id="test", scan_page=n, printed_page=str(n), page_type="text",
        scan_quality="good", language="English", header=header,
        main_text=text, notes_text=notes, vocab=[], unclear_count=0, flags=[],
    )


PAGES = [
    page(1, "INTRODUCTION.\n\nThe translator's introduction.", header=None),
    page(2, "THE PLACE OF A.\n\nLESSON I.\n\nFirst lesson text.", header=None),
    page(3, "More of lesson one.\n\nThus ends the first Lesson.\n\nLESSON II.\n\nShort.",
         notes="* A footnote."),
    page(4, "( 12 )\n\nDEAR SIR, I have received your translation.", header=None),
    page(5, "LESSON V.\n\nA stray heading in the middle of lesson two."),
    page(6, "B STHANAM.\n\nThe second division opens without a lesson heading.", header=None),
]


@pytest.fixture(scope="module")
def structure():
    return build_structure(PAGES, TINY, TINY_FIXES)


def texts_by_page(structure):
    result = {}
    for record in structure.lesson_pages:
        result.setdefault(record.scan_page, []).append(record)
    return result


# --- finding lesson starts ------------------------------------------------

def test_every_lesson_is_found_in_order(structure):
    assert [(s.sthana, s.lesson, s.scan_page) for s in structure.starts] == [
        ("a", 1, 2), ("a", 2, 3), ("b", 1, 6),
    ]
    assert structure.missing == []


def test_heading_out_of_sequence_is_ignored_and_reported(structure):
    assert len(structure.ignored_headings) == 1
    assert "page 5: 'LESSON V.'" in structure.ignored_headings[0]
    [record] = texts_by_page(structure)[5]
    assert (record.sthana, record.lesson) == ("a", 2)


def test_missing_lesson_is_reported():
    result = build_structure(PAGES[:5], TINY, fixes={})
    assert result.missing == [("b", 1)]


def test_fix_must_match_the_page():
    with pytest.raises(ValueError, match="not found"):
        build_structure(PAGES, TINY, {("b", 1): (6, "NO SUCH LINE")})


def test_fix_must_be_in_order():
    with pytest.raises(ValueError, match="out of order"):
        build_structure(PAGES, TINY, {("b", 1): (2, "THE PLACE OF A.")})


# --- body pages and wrapper pages -------------------------------------------

def test_wrapper_pages_are_left_out(structure):
    assert structure.wrapper_pages == [1, 4]
    assert {1, 4}.isdisjoint(texts_by_page(structure))


def test_headerless_page_counts_as_body_when_it_opens_a_lesson(structure):
    assert {2, 6} <= set(texts_by_page(structure))


# --- splitting pages ---------------------------------------------------------

def test_title_lines_above_a_heading_belong_to_the_new_lesson(structure):
    [record] = texts_by_page(structure)[2]
    assert record.main_text.startswith("THE PLACE OF A.\n\nLESSON I.")


def test_page_is_split_where_a_lesson_starts(structure):
    first, second = texts_by_page(structure)[3]
    assert (first.lesson, second.lesson) == (1, 2)
    assert first.main_text.endswith("Thus ends the first Lesson.")
    assert second.main_text == "LESSON II.\n\nShort."
    assert second.lesson_title == "A-two"


def test_page_notes_go_to_one_piece_only(structure):
    first, second = texts_by_page(structure)[3]
    assert first.notes_text == "* A footnote."   # the longer piece
    assert second.notes_text == ""


def test_output_round_trip(tmp_path, structure):
    out = tmp_path / "lesson_pages.jsonl"
    write_outputs(structure, out, tmp_path / "structure.json")
    assert load_lesson_pages(out) == structure.lesson_pages


# --- tables and helpers ----------------------------------------------------

def test_roman_numerals_are_strict():
    assert ROMAN_VALUES["XIX"] == 19
    assert "IXX" not in ROMAN_VALUES
    assert to_roman(22) == "XXII"


def test_contents_table_matches_the_translation():
    assert [(s.key, len(s.lessons)) for s in STHANAS] == [
        ("sutra", 30), ("vimana", 8), ("sharira", 8),
        ("indriya", 12), ("nidana", 8), ("chikitsa", 22),
    ]
    lessons = {(s.key, n) for s in STHANAS for n in range(1, len(s.lessons) + 1)}
    assert set(LESSON_START_FIXES) <= lessons


def test_lesson_label():
    assert lesson_label("sutra", 1) == "Sutra Sthana, Lesson I (Dirghanjivitiya)"


# --- the real source (skipped when it has not been downloaded) -------------

@pytest.mark.skipif(not config.SOURCE_PATH.exists(), reason="source text not downloaded")
def test_real_source_structure():
    result = build_structure(ingest_file(config.SOURCE_PATH))
    assert result.missing == []
    assert result.ignored_headings == []
    first_pages = {(s.sthana, s.lesson): s.scan_page for s in result.starts}
    assert first_pages[("sutra", 1)] == 11
    assert first_pages[("vimana", 1)] == 590
    assert first_pages[("sharira", 1)] == 837     # no sthana heading in the text
    assert first_pages[("indriya", 1)] == 1107    # no lesson heading in the text
    assert first_pages[("nidana", 1)] == 1192
    assert first_pages[("chikitsa", 1)] == 1297
    # press opinions, a review and a contents abstract typed as "text"
    assert {226, 273, 1172, 1334} <= set(result.wrapper_pages)
    # body pages whose running header is garbled by OCR
    assert {621, 732, 827}.isdisjoint(result.wrapper_pages)
