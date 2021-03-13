import os

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
    ".rb": "Ruby",
    ".rust": "Rust",
    ".scala": "Scala"
}

SUPPORTED_LANGUAGES = list(EXTENSION_TO_LANGUAGE.values())

LANGUAGE_FILES = [file for lang in SUPPORTED_LANGUAGES for file in os.listdir(f'lang/{lang}')]
