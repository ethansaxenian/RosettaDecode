"""
Preliminary Features List:
    - count of each special character
    - percent of each special character from the total number of special characters
    - percent of all chars that are special
    -
    - most common line endings
    - most common words (use substrings.jsonl)
    - most common n-length substrings of words (use substrings.jsonl)
"""
import re
from collections import Counter
from typing import Callable

SPECIAL_CHARACTERS_MAPPING = {
    "'": 0,
    "~": 1,
    "`": 2,
    "!": 3,
    "@": 4,
    "#": 5,
    "$": 6,
    "%": 7,
    "^": 8,
    "&": 9,
    "*": 10,
    "(": 11,
    ")": 12,
    "-": 13,
    "+": 14,
    "=": 15,
    "[": 16,
    "]": 17,
    "{": 18,
    "}": 19,
    "|": 20,
    ";": 21,
    ":": 22,
    '"': 23,
    ",": 24,
    ".": 25,
    "<": 26,
    ">": 27,
    "/": 28,
    "?": 29,
    "\\": 30,
    "...": 31,
}


def remove_spaces(code: str) -> str:
    return re.sub(r"[\n\t\s]*", "", code)


def find_words(code: str) -> list[str]:
    return [word for word in re.findall(r'\w+', code) if not word.isdecimal()]


def find_special_characters(code: str) -> list[str]:
    return re.findall(r'[.]{3}|[^0-9a-zA-Z_ \n]', code)  # find all special characters and ellipses


def features_per_line(code: str, regex: Callable[[str], list[str]]) -> list[int]:
    return [len(regex(line)) for line in code.split("\n")]


def n_length_substrings(n: int, code: str) -> list[str]:
    return [code[i:i+n] for i in range(len(code) - (n-1)) if code[i:i+n].isalpha()]


def count_line_endings(code: str) -> Counter[str, int]:
    return Counter(re.findall(r'.(?=\n)', code))


def pct_specials(code: str) -> float:
    return len(find_special_characters(code)) / len(remove_spaces(code))


if __name__ == '__main__':
    with open('lang/Haskell/24-game.hs', 'r') as file:
        code = file.read().lower()
        print(SPECIAL_CHARACTERS_MAPPING)
