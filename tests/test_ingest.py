from pathlib import Path

import pytest

from src.ingest import (
    clean_text,
    content_sha256,
    ingest_file,
    is_indexable,
    load_pages,
    split_main_and_notes,
    split_pages,
    write_jsonl,
)

FIXTURE = Path(__file__).parent / "fixtures" / "mini_source.txt"


@pytest.fixture(scope="module")
def pages():
    return {p.scan_page: p for p in ingest_file(FIXTURE)}


# --- splitting the file into pages ---------------------------------------

def test_split_pages_finds_every_page_and_drops_header_and_footer():
    pages = split_pages(FIXTURE.read_text(encoding="utf-8"))
    assert [n for n, _ in pages] == [1, 2, 3, 4, 5]
    all_text = "\n".join(body for _, body in pages)
    assert "Downloaded" not in all_text
    assert "END OF DOCUMENT" not in all_text
    assert "────" not in all_text


@pytest.mark.parametrize("text", ["", "plain text with no page markers"])
def test_split_pages_rejects_input_without_markers(text):
    with pytest.raises(ValueError):
        split_pages(text)


def test_split_pages_rejects_duplicate_markers():
    with pytest.raises(ValueError, match="duplicate"):
        split_pages("[Page 1]\na\n[Page 1]\nb\n")


# --- page metadata ----------------------------------------------------------

def test_metadata_is_read_from_tags(pages):
    page = pages[2]
    assert page.page_type == "text"
    assert page.printed_page == "1"
    assert page.scan_quality == "good"
    assert page.header == "CHARAKA-SAMHITĀ."
    assert page.vocab == ["Ayu", "pitta", "wind", "bile"]


def test_missing_values_become_none(pages):
    assert pages[1].printed_page is None   # "N/A" in the source
    assert pages[4].printed_page is None   # no <page-num> tag at all
    assert pages[3].vocab == []            # empty <vocab></vocab>


# --- cleaning ---------------------------------------------------------------

def test_markup_is_removed_but_words_are_kept(pages):
    main = pages[2].main_text
    assert "Longevity" in main
    assert "Oil subjugates wind" in main     # <unclear>wind</unclear> keeps its word
    for leftover in ("<", ">", "*", "#", "---", "†"):
        assert leftover not in main
    assert "A divider" not in main            # image descriptions are dropped


def test_clean_text_handles_centring_bold_tables_and_empty_brackets():
    assert clean_text("->**CALCUTTA**<-") == "CALCUTTA"
    assert clean_text("| Oil | Ghee |\n|---|---|") == "Oil Ghee"
    assert clean_text("alcohol.(*)") == "alcohol."


# --- main text vs. footnotes ----------------------------------------------

def test_rule_before_heading_does_not_start_footnotes(pages):
    main = pages[2].main_text
    assert "THE PLACE OF APHORISMS." in main
    assert "LESSON I." in main


def test_footnotes_and_side_notes_go_to_notes(pages):
    page = pages[2]
    assert "Longevity is Ayu in the original.—T." in page.notes_text
    assert "Bile is pitta.—T." in page.notes_text
    assert "a. Vide note b in p. 8." in page.notes_text   # <margin>
    assert "12" not in page.notes_text                    # bare number <note> dropped
    assert "Longevity is Ayu" not in page.main_text


def test_page_without_rule_falls_back_to_footnote_mark(pages):
    page = pages[4]
    assert page.main_text.startswith("Honey subjugates phlegm")
    assert "footnote with no rule" in page.notes_text
    assert "notes_split_by_marker" in page.flags


def test_italic_line_is_not_mistaken_for_a_footnote():
    body = "Main text.\n*(Here is a verse containing a summary.)*\nMore main text."
    main, notes, method = split_main_and_notes(body)
    assert (notes, method) == ("", "none")


# --- quality flags ----------------------------------------------------------

def test_flags(pages):
    assert pages[1].flags == ["low_scan_quality"]          # title page: no text-page flags
    assert pages[2].flags == ["has_unclear"]
    assert pages[2].unclear_count == 1
    assert "missing_printed_page" in pages[4].flags
    assert "empty" in pages[5].flags
    assert pages[3].flags == []                            # blank pages are not "empty text"


def test_only_real_text_pages_are_indexable(pages):
    assert [n for n, p in pages.items() if is_indexable(p)] == [2, 4]


# --- checksum ---------------------------------------------------------------

def test_checksum_ignores_download_header_and_line_endings():
    original = FIXTURE.read_bytes().replace(b"\r\n", b"\n")  # however Git checked it out
    redownloaded = original.replace(b"2026-01-01", b"2027-05-05").replace(b"\n", b"\r\n")
    assert content_sha256(redownloaded) == content_sha256(original)


def test_checksum_changes_when_page_content_changes():
    original = FIXTURE.read_bytes()
    edited = original.replace(b"Honey subjugates", b"Honey increases")
    assert content_sha256(edited) != content_sha256(original)


def test_checksum_rejects_wrong_file():
    with pytest.raises(ValueError):
        content_sha256(b"")


# --- output -----------------------------------------------------------------

def test_jsonl_round_trip(tmp_path, pages):
    out = tmp_path / "pages.jsonl"
    write_jsonl(list(pages.values()), out)
    assert load_pages(out) == list(pages.values())
