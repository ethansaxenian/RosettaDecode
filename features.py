"""
Preliminary Features List:
    - count of each special character
    - percent of each special character out of the total number of special characters
    - percent of all chars that are special
    - most common line endings (use all characters mapping or unicode with ord())
    - most common words (use vocabulary.jsonl)
    - n-grams

Bag-of-words model?
    - use reserved words in the languages?
    - use special characters?
"""
import json
import os
import re
from collections import Counter
from typing import Callable

from file_path_storer import generate_file_paths
from language_info import get_language_from_filename, RESERVED_KEYWORDS

SPECIAL_CHAR_NAMES = {"'": "squote", "~": "tilde", "`": "backtick", "!": "exclaim", "@": "at", "#": "pound",
                      "$": "dollar", "%": "pct", "^": "caret", "&": "amp", "*": "times", "(": "lparen",
                      ")": "rparen", "-": "minus", "+": "plus", "=": "eq", "[": "lbracket", "]": "rbracket",
                      "{": "lbrace", "}": "rbrace", "|": "pipe", ";": "semicolon", ":": "colon", '"': "dquote",
                      ",": "comma", ".": "dot", "<": "langle", ">": "rangle", "/": "fslash", "?": "question",
                      "\\": "bslash", "...": "ellipsis"}

SPECIALS = list(SPECIAL_CHAR_NAMES.keys())

ALL_CHAR_MAPPING = {"'": 0, '~': 1, '`': 2, '!': 3, '@': 4, '#': 5, '$': 6, '%': 7, '^': 8, '&': 9, '*': 10, '(': 11,
                    ')': 12, '-': 13, '+': 14, '=': 15, '[': 16, ']': 17, '{': 18, '}': 19, '|': 20, ';': 21, ':': 22,
                    '"': 23, ',': 24, '.': 25, '<': 26, '>': 27, '/': 28, '?': 29, '\\': 30, '...': 31, 'a': 32,
                    'b': 33, 'c': 34, 'd': 35, 'e': 36, 'f': 37, 'g': 38, 'h': 39, 'i': 40, 'j': 41, 'k': 42, 'l': 43,
                    'm': 44, 'n': 45, 'o': 46, 'p': 47, 'q': 48, 'r': 49, 's': 50, 't': 51, 'u': 52, 'v': 53, 'w': 54,
                    'x': 55, 'y': 56, 'z': 57, '_': 58, '1': 59, '2': 60, '3': 61, '4': 62, '5': 63, '6': 64, '7': 65,
                    '8': 66, '9': 67, '0': 68}


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
    return Counter(re.findall(r'[.]{3}|.(?=\n)', re.sub(r"\s*", "", code)))


def pct_specials(code: str) -> float:
    return len(find_special_characters(code)) / len(remove_spaces(code))


def generate_feature_set(code: str) -> dict[str: int]:
    feature_set = {}

    specials_count = Counter(find_special_characters(code))

    bag_of_words = Counter(find_words(code))
    for keyword in RESERVED_KEYWORDS:
        feature_set[keyword] = bag_of_words[keyword]

    num_specials = len(list(specials_count.elements()))
    for char in SPECIALS:
        feature_set[f'num_{SPECIAL_CHAR_NAMES[char]}'] = specials_count[char]
        percent_specials = (specials_count[char] / num_specials) if num_specials > 0 else 0
        feature_set[f'percent_{SPECIAL_CHAR_NAMES[char]}'] = percent_specials

    feature_set['percent_specials'] = pct_specials(code)

    for rank, (char, _) in enumerate(count_line_endings(code).most_common(5)):
        feature_set[f'line_ending_{rank+1}'] = ALL_CHAR_MAPPING[char]

    return feature_set


def compile_dataset():
    with open("data/features_data.jsonl", "w") as outfile, open("data/file_paths.jsonl", "r") as infile:
        for line in infile:
            info = json.loads(line)
            with open(info["path"], "r") as file:
                code = file.read().lower()
                data = {
                    "name": os.path.basename(info["path"]),
                    "features": generate_feature_set(code),
                    "lang": get_language_from_filename(info["path"]),
                    # "code": code,
                }
                json.dump(data, outfile)
                outfile.write("\n")


if __name__ == '__main__':
    generate_file_paths()
    compile_dataset()
    # with open("lang/Python/24-game-1.py", "r") as file:
    #     code = file.read().lower()
    #     print(sorted(find_words(code)))
    # print(len(RESERVED_KEYWORDS))
