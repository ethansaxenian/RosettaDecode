"""
Preliminary Features List:
    - count of each special character
    - percent of each special character out of the total number of special characters
    - percent of all chars that are special
    - most common line endings (use all characters mapping or unicode with ord())
    - most common words (use vocabulary.jsonl)
    - n-grams?

Bag-of-words model
    - use reserved words in the languages
    - use special characters
"""
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Callable

import unidecode

from data_wrangling.file_path_storer import generate_file_paths
from data_wrangling.language_info import get_language_from_filename

SPECIAL_CHAR_NAMES = {"'": "squote", "~": "tilde", "`": "backtick", "!": "exclaim", "@": "at", "#": "pound",
                      "$": "dollar", "%": "pct", "^": "caret", "&": "amp", "*": "times", "(": "lparen",
                      ")": "rparen", "-": "minus", "+": "plus", "=": "eq", "[": "lbracket", "]": "rbracket",
                      "{": "lbrace", "}": "rbrace", "|": "pipe", ";": "semicolon", ":": "colon", '"': "dquote",
                      ",": "comma", ".": "dot", "<": "langle", ">": "rangle", "/": "fslash", "?": "question",
                      "\\": "bslash", "...": "ellipsis"}

SPECIALS = list(SPECIAL_CHAR_NAMES.keys())

CHAR_MAPPING = {"'": 0, '~': 1, '`': 2, '!': 3, '@': 4, '#': 5, '$': 6, '%': 7, '^': 8, '&': 9, '*': 10, '(': 11,
                ')': 12, '-': 13, '+': 14, '=': 15, '[': 16, ']': 17, '{': 18, '}': 19, '|': 20, ';': 21, ':': 22,
                '"': 23, ',': 24, '.': 25, '<': 26, '>': 27, '/': 28, '?': 29, '\\': 30, '...': 31, 'a': 32, 'b': 33,
                'c': 34, 'd': 35, 'e': 36, 'f': 37, 'g': 38, 'h': 39, 'i': 40, 'j': 41, 'k': 42, 'l': 43, 'm': 44,
                'n': 45, 'o': 46, 'p': 47, 'q': 48, 'r': 49, 's': 50, 't': 51, 'u': 52, 'v': 53, 'w': 54, 'x': 55,
                'y': 56, 'z': 57, '_': 58, '1': 59, '2': 60, '3': 61, '4': 62, '5': 63, '6': 64, '7': 65, '8': 66,
                '9': 67, '0': 68}

RESERVED_KEYWORDS = ['__end__', 'and', 'any', 'as', 'assert', 'auto', 'begin', 'bool', 'boolean', 'break', 'byte',
                     'case', 'catch', 'char', 'check', 'class', 'const', 'continue', 'cout', 'data', 'def', 'default',
                     'delete', 'deriving', 'do', 'double', 'elif', 'else', 'elseif', 'elsif', 'end', 'endl', 'error',
                     'eval', 'except', 'export', 'extends', 'extern', 'false', 'final', 'float64', 'for', 'foreach',
                     'from', 'func', 'function', 'go', 'goto', 'if', 'implements', 'import', 'in', 'include',
                     'instance', 'int', 'int64', 'interface', 'iostream', 'is', 'lambda', 'last', 'let', 'local',
                     'long', 'main', 'map', 'module', 'my', 'namespace', 'new', 'next', 'nil', 'none', 'not', 'nothing',
                     'null', 'of', 'operator', 'or', 'our', 'package', 'print', 'private', 'public', 'qualified',
                     'raise', 'range', 'return', 'self', 'sizeof', 'static', 'std', 'string', 'struct', 'switch',
                     'template', 'then', 'this', 'thread', 'throw', 'throws', 'true', 'try', 'type', 'typedef',
                     'typename', 'typeof', 'undef', 'union', 'unless', 'unsigned', 'use', 'using', 'var', 'void',
                     'when', 'where', 'while', 'with']


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
    return Counter(re.findall(r'.(?=\n)', re.sub(r"[ |\t]*", "", code)))


def pct_specials(code: str) -> float:
    return len(find_special_characters(code)) / len(remove_spaces(code))


def extract_features(code: str, binary_counts: bool = False, keywords: list[str] = None) -> dict[str: int]:
    """
    returns a dictionary of features extracted from a string
    """
    if keywords is None:
        keywords = RESERVED_KEYWORDS

    features_dict = {}

    bag_of_words = Counter(find_words(code))
    for keyword in keywords:
        num = bag_of_words[keyword]
        features_dict[keyword] = int(bool(num)) if binary_counts else num

    specials_count = Counter(find_special_characters(code))
    num_specials = len(list(specials_count.elements()))
    for char in SPECIALS:
        features_dict[f'num_{SPECIAL_CHAR_NAMES[char]}'] = specials_count[char]
        percent_specials = (specials_count[char] / num_specials) if num_specials > 0 else 0
        features_dict[f'percent_{SPECIAL_CHAR_NAMES[char]}'] = percent_specials

    most_common_ending = count_line_endings(code).most_common(1)[0][0]
    features_dict['most_frequent_line_ending'] = CHAR_MAPPING[most_common_ending]

    return features_dict


def compile_dataset(filename: str, lowercase: bool = True, binary_counts: bool = False, keywords: list[str] = None):
    """
    stores the features data for each code file in data/features_data.jsonl
    """
    with open(f"../data/{filename}{'_bc' if binary_counts else ''}.jsonl", "w") as outfile, open(
            "../data/file_paths.jsonl", "r") as infile:
        for line in infile:
            info = json.loads(line)
            data = parse_file(info["path"], lowercase, binary_counts, keywords)
            json.dump(data, outfile)
            outfile.write("\n")


def read_file(path: str, lowercase: bool = True) -> str:
    """
    reads a file path and returns the contents as a string
    """
    with open(path, "r") as file:
        code = file.read()
        if lowercase:
            code = code.lower()
        return unidecode.unidecode(code)


def parse_file(path: str, lowercase: bool = True, binary_counts: bool = False, keywords: list[str] = None) -> dict[str: int]:
    """
    compiles and returns the data from a single file path, which includes the filename, the features dictionary,
    and the language of the code in the file
    """
    code = read_file(path, lowercase)
    return {
        "name": os.path.basename(path),
        "features": extract_features(code, binary_counts, keywords),
        "lang": get_language_from_filename(path),
        # "code": code,
    }


if __name__ == '__main__':
    if not Path("../data/features_data.jsonl").exists():
        generate_file_paths()
    compile_dataset("features_data")
