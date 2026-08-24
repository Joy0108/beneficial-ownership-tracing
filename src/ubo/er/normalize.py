"""Normalisation: transliteration, legal forms, addresses, phonetics.

Everything here is reversible in the sense that the original string is kept on
the record. Normalisation feeds blocking and scoring; it never overwrites what
the register said, because a memo has to quote the register.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# Romanisation of the Cyrillic and Arabic forms that appear in sanctions data.
# Deliberately a table rather than a library call: sanctions screening has to be
# able to explain why two names were treated as the same string.
_CYRILLIC = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z",
    "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh",
    "щ": "shch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    "і": "i", "ї": "yi", "є": "ye", "ґ": "g",
}
_ARABIC = {
    "ا": "a", "ب": "b", "ت": "t", "ث": "th", "ج": "j", "ح": "h", "خ": "kh", "د": "d", "ذ": "dh",
    "ر": "r", "ز": "z", "س": "s", "ش": "sh", "ص": "s", "ض": "d", "ط": "t", "ظ": "z", "ع": "a",
    "غ": "gh", "ف": "f", "ق": "q", "ك": "k", "ل": "l", "م": "m", "ن": "n", "ه": "h", "و": "w",
    "ي": "y", "ى": "a", "ء": "", "أ": "a", "إ": "i", "آ": "a", "ة": "h", "ّ": "",
}

# Legal forms across the jurisdictions in the corpus. Stripping these is what
# lets "Northwind Energy Trading Ltd" and "NORTHWIND ENERGY TRADING LIMITED"
# block together; keeping them would make the suffix dominate short names.
#
# The list holds only tokens that denote a *legal form*. Words like "Group", "Holdings" and
# "International" look like boilerplate and are not: stripping them collapses
# "Regent Ventures Group" and "Regent Ventures Enterprises" onto each other, and
# two unrelated firms sharing a trading stem is the hardest negative there is.
_LEGAL_FORMS = {
    "ltd", "limited", "llc", "lc", "plc", "llp", "lp", "inc", "incorporated", "corp", "corporation",
    "co", "company", "gmbh", "ag", "sa", "sas", "sarl", "bv", "nv", "spa", "srl", "as", "ab", "oy",
    "aps", "pte", "pty", "kk", "ooo", "oao", "pao", "zao", "ibc", "fze", "fzc", "fzco", "dmcc",
    "sociedad", "anonima", "aktiengesellschaft", "besloten", "vennootschap", "exempted", "bvi",
    "business", "public", "free", "zone", "establishment",
}

# Address noise words. "suite 3, second floor" is not evidence of anything.
_ADDRESS_STOP = {
    "suite", "floor", "unit", "office", "po", "box", "level", "room", "building", "house", "tower",
    "street", "st", "road", "rd", "avenue", "ave", "lane", "drive", "the", "of", "and", "c/o",
}

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_WS = re.compile(r"\s+")


@lru_cache(maxsize=8192)
def transliterate(text: str) -> str:
    """Romanise, then fold accents. Idempotent on Latin input."""
    out = []
    for ch in text:
        low = ch.lower()
        if low in _CYRILLIC:
            out.append(_CYRILLIC[low])
        elif low in _ARABIC:
            out.append(_ARABIC[low])
        else:
            out.append(ch)
    folded = unicodedata.normalize("NFD", "".join(out))
    return "".join(c for c in folded if unicodedata.category(c) != "Mn")


def _merge_initialisms(tokens: list[str]) -> list[str]:
    """Rejoin runs of single characters: p l c -> plc, s p a -> spa.

    Punctuated legal forms are written both ways in every register - "P.L.C."
    and "PLC", "S.p.A." and "SpA" - and splitting on punctuation turns one of
    them into three meaningless one-character tokens that no legal-form list
    will ever match.
    """
    out: list[str] = []
    run: list[str] = []
    for token in tokens:
        if len(token) == 1 and token.isalpha():
            run.append(token)
            continue
        if len(run) >= 2:
            out.append("".join(run))
        elif run:
            out.extend(run)
        run = []
        out.append(token)
    if len(run) >= 2:
        out.append("".join(run))
    elif run:
        out.extend(run)
    return out


@lru_cache(maxsize=8192)
def normalize_name(name: str, drop_legal_forms: bool = True) -> str:
    text = transliterate(name).lower()
    text = _NON_ALNUM.sub(" ", text)
    tokens = _merge_initialisms([t for t in text.split() if t])
    if drop_legal_forms:
        stripped = [t for t in tokens if t not in _LEGAL_FORMS]
        # Never strip a name down to nothing: "Holdings Limited" is a real name.
        tokens = stripped or tokens
    return " ".join(tokens)


def name_tokens(name: str) -> list[str]:
    return normalize_name(name).split()


def significant_tokens(name: str, min_length: int = 3) -> list[str]:
    return [t for t in name_tokens(name) if len(t) >= min_length]


@lru_cache(maxsize=8192)
def normalize_address(address: str) -> str:
    text = transliterate(address).lower()
    text = _NON_ALNUM.sub(" ", text)
    return " ".join(t for t in text.split() if t not in _ADDRESS_STOP and not t.isdigit())


def address_tokens(address: str) -> set[str]:
    return set(normalize_address(address).split())


# --- phonetics -------------------------------------------------------------
#
# A Soundex variant. Its job here is narrow: survive the vowel and voicing
# differences that transliteration introduces (Kuznetsova / Kouznetsova,
# Morozov / Morosov) without collapsing genuinely different names.

# 'w' is grouped with 'v' rather than ignored, as standard Soundex would.
# Romanisation of Slavic and Germanic names alternates the two freely - Volkov
# and Wolkow are the same name - and standard Soundex puts them in different
# blocks, which is a miss no downstream stage can recover from.
_SOUNDEX_CODES = {
    **dict.fromkeys("bfpvw", "1"),
    **dict.fromkeys("cgjkqsxz", "2"),
    **dict.fromkeys("dt", "3"),
    "l": "4",
    **dict.fromkeys("mn", "5"),
    "r": "6",
}


# Standard Soundex keeps the initial letter verbatim, which defeats the whole
# purpose here: Volkov and Wolkow agree on every following code and still land in
# different blocks because one starts V and the other W. These are the initial
# substitutions that romanisation actually makes.
_FIRST_LETTER_FOLD = {"w": "v", "c": "k", "y": "i", "j": "i", "q": "k", "x": "s"}


@lru_cache(maxsize=8192)
def soundex(token: str) -> str:
    text = transliterate(token).lower()
    text = "".join(c for c in text if c.isalpha())
    if not text:
        return ""
    first, rest = _FIRST_LETTER_FOLD.get(text[0], text[0]), text[1:]
    codes = [_SOUNDEX_CODES.get(first, "")]
    for ch in rest:
        code = _SOUNDEX_CODES.get(ch, "")
        if code and code != codes[-1]:
            codes.append(code)
        elif not code and ch != "h":
            codes.append("")
    digits = "".join(c for c in codes[1:] if c)
    return (first.upper() + digits + "000")[:4]


def phonetic_key(name: str) -> str:
    """Soundex of every significant token, sorted so word order does not matter."""
    tokens = significant_tokens(name)
    return "-".join(sorted(soundex(t) for t in tokens if soundex(t)))


def normalize_birth_date(value: str) -> str:
    """Keep whatever precision the register gave. PSC publishes year-month only."""
    value = (value or "").strip()
    match = re.match(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", value)
    if not match:
        return ""
    year, month, day = match.groups()
    return "-".join(p for p in (year, month, day) if p)


def normalize_jurisdiction(value: str) -> str:
    return (value or "").strip().upper()[:2]


def normalize_identifier(value: str) -> str:
    return _NON_ALNUM.sub("", (value or "").lower())
