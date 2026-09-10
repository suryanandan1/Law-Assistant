from src.chunker import DocumentChunker, find_headings, heading_before

# ----------------------------------
# create_chunk_id
# ----------------------------------


def test_chunk_id_is_stable_for_same_input():
    a = DocumentChunker.create_chunk_id(21, "Protection of life and personal liberty")
    b = DocumentChunker.create_chunk_id(21, "Protection of life and personal liberty")
    assert a == b
    assert len(a) == 24


def test_chunk_id_changes_with_page():
    assert DocumentChunker.create_chunk_id(21, "same text") != DocumentChunker.create_chunk_id(
        22, "same text"
    )


def test_chunk_id_changes_with_text():
    assert DocumentChunker.create_chunk_id(21, "text one") != DocumentChunker.create_chunk_id(
        21, "text two"
    )


# ----------------------------------
# find_headings / heading_before
# ----------------------------------


def test_find_headings_detects_titled_sections():
    text = "21A. Right to education. The State shall. 22. Protection against arrest and detention. No person."
    kinds = {(k, i, t) for _, k, i, t in find_headings(text)}
    assert ("section", "21A", "Right to education") in kinds
    assert ("section", "22", "Protection against arrest and detention") in kinds


def test_find_headings_rejects_allcaps_banner():
    text = "447. CHAPTER XV COMPROMISES AND AMALGAMATIONS 230. Power to compromise with creditors."
    sections = [i for _, k, i, _ in find_headings(text) if k == "section"]
    assert "447" not in sections  # ALL-CAPS pseudo-title rejected
    assert "230" in sections


def test_find_headings_detects_article_and_part():
    found = find_headings("PART III deals with Article 14 and Article 21 of the document.")
    assert any(k == "part" and i == "III" for _, k, i, _ in found)
    assert any(k == "article" and i == "14" for _, k, i, _ in found)


def test_heading_before_returns_nearest_preceding():
    headings = [
        (0, "section", "1", "First"),
        (50, "section", "2", "Second"),
        (100, "section", "3", "Third"),
    ]
    assert heading_before(60, headings) == ("section", "2", "Second")
    assert heading_before(0, headings) == ("section", "1", "First")


def test_heading_before_returns_none_when_offset_precedes_all():
    assert heading_before(10, [(50, "section", "2", "Second")]) is None
