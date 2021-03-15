import re
from collections import Counter
from typing import Callable

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
SPECIAL_CHARACTERS = '~`!@#$%^&*()_-+=[]{}|;:",.<>/?\\\''


def is_word(string: str) -> bool:
    return all(c in ALPHABET for c in string)


def remove_spaces(string: str) -> str:
    return re.sub(r"[\n\t\s]*", "", string)


def count_words(code: str) -> Counter[str, int]:
    words = [word for word in re.findall(r'\w+', code) if not word.isdecimal()]
    return Counter(words)


def count_special_characters(code: str) -> Counter[str, int]:
    specials = re.findall(r'[.]{3}|[^0-9a-zA-Z_ \n]', code)  # find all special characters and ellipses
    return Counter(specials)


def count_line_endings(code: str) -> Counter[str, int]:
    endings = re.findall(r'.(?=\n)', code)
    return Counter(endings)


def features_per_line(code: str, regex: Callable[[str], list[str]]) -> list[int]:
    counts = []
    for line in code.split("\n"):
        counts.append(len(regex(line)))
    return counts


def count_n_length_substrings(n: int, code: str) -> Counter[str, int]:
    substrings = [code[i:i+n] for i in range(len(code) - (n-1)) if code[i:i+n].isalpha()]
    return Counter(substrings)


if __name__ == '__main__':
    with open('lang/Python/a+b-1.py', 'r') as file:
        code = file.read().lower()

