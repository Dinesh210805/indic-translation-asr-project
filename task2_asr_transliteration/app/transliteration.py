import logging
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate as _transliterate

logger = logging.getLogger(__name__)

SCHEME_MAP = {
    "ITRANS": sanscript.ITRANS,
    "ISO": sanscript.ISO,
    "IAST": sanscript.IAST,
    "HK": sanscript.HK,
    "SLP1": sanscript.SLP1,
}


def transliterate_tamil_to_latin(text: str, scheme: str = "ITRANS") -> str:
    """Convert Tamil Unicode text to Roman script using the given scheme.

    Non-Tamil characters (e.g., English loanwords) pass through unchanged.
    Returns original text as fallback on any transliteration error.
    """
    if not text or not text.strip():
        return ""

    if scheme not in SCHEME_MAP:
        raise ValueError(f"Unknown scheme: {scheme}. Valid: {list(SCHEME_MAP.keys())}")

    try:
        return _transliterate(text, sanscript.TAMIL, SCHEME_MAP[scheme])
    except Exception as exc:
        logger.error("Transliteration failed for scheme %s: %s", scheme, exc)
        return text


def transliterate_latin_to_tamil(text: str, scheme: str = "ITRANS") -> str:
    """Convert romanized text back to Tamil Unicode script."""
    if not text or not text.strip():
        return ""

    if scheme not in SCHEME_MAP:
        raise ValueError(f"Unknown scheme: {scheme}. Valid: {list(SCHEME_MAP.keys())}")

    try:
        return _transliterate(text, SCHEME_MAP[scheme], sanscript.TAMIL)
    except Exception as exc:
        logger.error("Reverse transliteration failed: %s", exc)
        return text
