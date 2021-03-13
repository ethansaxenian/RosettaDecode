"""
Potential features:
    - most frequent words
    - most frequent special characters
    - most frequent line endings
    - avg line length / words per line
"""
import re
from collections import Counter

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
SPECIAL_CHARACTERS = '~`!@#$%^&*()_-+=[]{}|;:",.<>/?\\\''


def count_words(code: str) -> dict[str, int]:
    words = [word for word in re.findall(r'\w+', code) if not word.isdecimal()]
    return Counter(words)


def count_special_characters(code: str) -> dict[str, int]:
    specials = [char for char in code if char in SPECIAL_CHARACTERS]
    return Counter(specials)


def count_line_endings(code: str) -> dict[str, int]:
    endings = []
    for line in code.split("\n"):
        try:
            endings.append(line.strip()[-1])
        except IndexError:
            continue
    return Counter(endings)


if __name__ == '__main__':
    with open('lang/Python/conways-game-of-life-3.py', 'r') as file:
        code = file.read().lower()
        print(count_line_endings(code))
        print(count_words(code))
        print(count_special_characters(code))

