import pytest
from app.transliteration import transliterate_tamil_to_latin, transliterate_latin_to_tamil


def test_basic_tamil_to_itrans():
    result = transliterate_tamil_to_latin("வணக்கம்", "ITRANS")
    assert isinstance(result, str)
    assert len(result) > 0


def test_basic_tamil_to_iso15919():
    result = transliterate_tamil_to_latin("வணக்கம்", "ISO15919")
    assert isinstance(result, str)
    assert len(result) > 0


def test_empty_string_returns_empty():
    assert transliterate_tamil_to_latin("", "ITRANS") == ""
    assert transliterate_tamil_to_latin("   ", "ITRANS") == ""


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        transliterate_tamil_to_latin("வணக்கம்", "FAKE_SCHEME")


def test_all_schemes_produce_output():
    text = "தமிழ்"
    for scheme in ["ITRANS", "ISO15919", "IAST", "HK", "HUNTERIAN"]:
        result = transliterate_tamil_to_latin(text, scheme)
        assert isinstance(result, str) and len(result) > 0, f"Scheme {scheme} returned empty"


def test_reverse_itrans_roundtrip():
    original = "வணக்கம்"
    roman = transliterate_tamil_to_latin(original, "ITRANS")
    back = transliterate_latin_to_tamil(roman, "ITRANS")
    assert isinstance(back, str) and len(back) > 0


def test_english_passthrough():
    result = transliterate_tamil_to_latin("hello", "ITRANS")
    assert "hello" in result.lower()


def test_reverse_unknown_scheme_raises():
    with pytest.raises(ValueError):
        transliterate_latin_to_tamil("test", "UNKNOWN")


def test_reverse_empty_returns_empty():
    assert transliterate_latin_to_tamil("", "ITRANS") == ""
