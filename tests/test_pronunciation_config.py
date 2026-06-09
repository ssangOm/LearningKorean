from pathlib import Path

from src.pronunciation.config import DATA_DIR, OUTPUT_DIR, SENTENCES


def test_sentence_ids_are_unique():
    sentence_ids = [sentence.sentence_id for sentence in SENTENCES]

    assert len(sentence_ids) == len(set(sentence_ids))


def test_sentences_have_korean_text():
    assert len(SENTENCES) == 10
    for sentence in SENTENCES:
        assert sentence.text.strip()
        assert any("\uac00" <= char <= "\ud7a3" for char in sentence.text)


def test_default_paths_are_project_local():
    assert DATA_DIR == Path("data/pronunciation")
    assert OUTPUT_DIR == Path("outputs/pronunciation")
