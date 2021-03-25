import os
import pathlib
from enum import Enum


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

LANGUAGE_FILES = [file for lang in EXTENSION_TO_LANGUAGE.values() for file in os.listdir(f'lang/{lang}')]


def get_language_from_filename(filename: str) -> str:
    ext = pathlib.Path(filename).suffix
    return EXTENSION_TO_LANGUAGE[ext]
