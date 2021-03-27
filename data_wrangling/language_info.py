import os
import pathlib


EXT_TO_LANG = {
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

LANG_FILES = [file for lang in EXT_TO_LANG.values() for file in os.listdir(f'../lang/{lang}')]


def get_language_from_filename(filename: str) -> str:
    ext = pathlib.Path(filename).suffix
    return EXT_TO_LANG[ext]
