import os
from enum import Enum


class Language(Enum):
    C = "C"
    CPP = "C++"
    Go = "Go"
    Haskell = "Haskell"
    Java = "Java"
    JavaScript = "JavaScript"
    Julia = "Julia"
    Perl = "Perl"
    Python = "Python"
    Ruby = "Ruby"


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

LANGUAGE_FILES = [file for lang in Language for file in os.listdir(f'lang/{lang.value}')]
