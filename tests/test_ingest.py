from mhrag.ingest import clean, pdf, qa
from mhrag.ingest.chunk import chunk_document, get_token_counter
from mhrag.ingest.corpus import load_documents
from mhrag.ingest.dedup import drop_near_duplicates, near_duplicates
from mhrag.ingest.sources import RawData
from mhrag.types import Chunk, Document, Section
from tests.fixtures import ANGER_URL, INTRO


def _docs(settings):
    return load_documents(RawData(settings.paths.raw_data), settings.ingest)


def test_pdf_header_becomes_metadata_and_is_stripped(settings):
    corpus = _docs(settings)
    anger = next(d for d in corpus.documents if d.source_file.endswith("anger.pdf"))
    assert anger.url == ANGER_URL and anger.url_verified
    assert anger.title == "When is anger a problem?"
    assert anger.source_type == "mind_web"
    text = anger.text
    assert "9:21 AM" not in text and "1/1" not in text
    assert "registered charity" not in text  # footer boilerplate
    assert "Gymraeg" not in text and "Welsh" not in text  # Welsh-translation notice


def test_question_headings_become_sections(settings):
    anger = next(d for d in _docs(settings).documents if d.source_file.endswith("anger.pdf"))
    headings = [s.heading for s in anger.sections]
    assert "What is anger?" in headings and "How can anger be helpful?" in headings


def test_repeated_intro_kept_once(settings):
    docs = [d for d in _docs(settings).documents if d.source_type == "mind_web"]
    assert sum(INTRO[:40] in d.text for d in docs) == 1


def test_image_only_pdf_skipped_and_reported(settings):
    stats = _docs(settings).stats
    assert any("easy-read" in s for s in stats["pdf"]["skipped_no_text_layer"])


def test_kb_facts_only_and_known_bad_intent_dropped(settings):
    corpus = _docs(settings)
    kb = [d for d in corpus.documents if d.source_type == "kb_fact"]
    assert [d.meta["tag"] for d in kb] == ["fact-1"]
    assert "fact-10" in corpus.stats["kb"]["dropped_intents"]


def test_stringified_response_list_is_parsed(settings):
    intents = qa.load_intents(RawData(settings.paths.raw_data), qa.KB_FILE)
    scared = next(i for i in intents if i.tag == "scared")
    assert scared.responses == ["It is natural to feel scared.", "Take a breath."]


def test_faq_loaded_and_duplicate_file_reported(settings):
    corpus = _docs(settings)
    assert sum(d.source_type == "faq" for d in corpus.documents) == 2
    assert corpus.stats["faq"]["mentalhealth_csv_rows_in_faq"] == 2


def test_counselling_off_by_default_and_split_deterministic(settings):
    assert not any(d.source_type == "counselling" for d in _docs(settings).documents)
    assert qa.counselling_split("hello") == qa.counselling_split("hello ")


def test_exclude_faq_from_index(settings):
    settings.ingest.exclude_faq_from_index = True
    assert not any(d.source_type == "faq" for d in _docs(settings).documents)


def test_normalize_unicode_fixes_ligatures():
    assert clean.normalize_unicode("diﬀerent diﬃcult ﬁne self￾harm") == "different difficult fine self-harm"


def test_parse_pdf_booklet():
    pages = ["© Mind 2020\n1\nSuicidal feelings\nExplains what suicidal feelings are.\nContents\n"
             "What are suicidal feelings? ........ 2\n",
             "© Mind 2020\n2\nWhat are suicidal feelings?\nSuicide is the act of intentionally taking your own life."]
    r = pdf.parse_pdf("PDF_Files/suicidal-feelings.pdf", pages)
    assert r.source_type == "mind_booklet" and r.year == 2020 and r.title == "Suicidal feelings"
    assert "what are suicidal feelings" in r.known_headings
    assert not any(ln.startswith("© Mind") for ln in r.lines)


def test_chunks_respect_token_limit_and_keep_metadata():
    count = get_token_counter(None)
    body = " ".join(f"Sentence number {i} talks about sleep hygiene and routines." for i in range(80))
    doc = Document("d1", "Sleep", "f.pdf", "mind_web", [Section("How can I sleep better?", body)], url="u")
    chunks = chunk_document(doc, 64, 0.15, count)
    assert len(chunks) > 3
    assert all(c.n_tokens <= 64 - 16 for c in chunks)  # 16 tokens reserved for the title/section header
    assert all(count(c.embed_text) <= 64 + 4 for c in chunks)
    assert all(c.section == "How can I sleep better?" and c.url == "u" for c in chunks)
    # overlap: consecutive chunks share text
    assert chunks[0].text.split(". ")[-1][:20] in chunks[1].text


def test_near_duplicates_detected_and_higher_priority_source_kept():
    a = "Anger is a normal healthy emotion that everyone feels at times and it can be useful in many situations"
    b = a + " too"
    assert near_duplicates([a, b, "something completely different about sleep"], 0.8)
    mk = lambda i, t, st: Chunk(i, i, t, "T", "S", ["S"], "f", st, None, n_tokens=20)  # noqa: E731
    kept, log = drop_near_duplicates([mk("x", b, "web_article"), mk("y", a, "mind_web")], 0.8)
    assert [c.chunk_id for c in kept] == ["y"] and log


def test_fix_mojibake():
    assert clean.fix_mojibake("personâ€™s health") == "person’s health"
    assert clean.fix_mojibake("already fine ’") == "already fine ’"
