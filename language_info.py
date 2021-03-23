import os
import pathlib
from enum import Enum


# class Language(Enum):
#     C = 0
#     CPP = 1
#     Go = 2
#     Haskell = 3
#     Java = 4
#     JavaScript = 5
#     Julia = 6
#     Perl = 7
#     Python = 8
#     Ruby = 9
#     # Other = 10  # is this needed?


EXTENSION_TO_LANGUAGE = {
    ".c": "C",
    ".cpp": "C++",
    ".go": "Go",
    ".hs": "Haskell",
    ".java": "Java",
    ".js": "JavaScript",
    ".julia": "Julia",
    ".pl": "Perl",
    ".py": "Python",
    ".rb": "Ruby"
}

LANG_TO_INT = {lang: i for (i, lang) in enumerate(EXTENSION_TO_LANGUAGE.values())}

LANGUAGE_FILES = [file for lang in EXTENSION_TO_LANGUAGE.values() for file in os.listdir(f'lang/{lang}')]


def get_language_from_filename(filename: str) -> str:
    ext = pathlib.Path(filename).suffix
    return EXTENSION_TO_LANGUAGE[ext]
