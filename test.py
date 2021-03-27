import glob
import os
import pathlib

EXTENSION_TO_LANGUAGE = {
    ".c": "C",
    ".cpp": "C++",
    ".go": "Go",
    ".hs": "Haskell",
    ".java": "Java",
    ".js": "JavaScript",
    ".jl": "Julia",
    ".pl": "Perl",
    ".py": "Python",
    ".rb": "Ruby"
}

for lang in ['julia']:
    file_list = [f for f in glob.iglob(f'{lang}-master/**', recursive=True) if os.path.isfile(f) and pathlib.Path(f).suffix in EXTENSION_TO_LANGUAGE.keys()]
    assert len(file_list) > 0
    os.makedirs(f"lang_extra/{lang}-files", exist_ok=True)

    for file in file_list:
        ext = pathlib.Path(file).suffix
        os.replace(file, f"lang_extra/{EXTENSION_TO_LANGUAGE[ext]}-files/{os.path.basename(file)}")
