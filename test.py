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

lang = "Haskell"
file_list = [f for f in glob.iglob(f'purescript-master/**', recursive=True) if os.path.isfile(f) and pathlib.Path(f).suffix in EXTENSION_TO_LANGUAGE.keys()]
assert len(file_list) > 0
# print(len(file_list))
# print(len([f for f in file_list if pathlib.Path(f).suffix == ".hs"]))

os.makedirs(f"lang2/{lang}", exist_ok=True)

for file in [f for f in file_list if pathlib.Path(f).suffix == ".hs"]:
    ext = pathlib.Path(file).suffix
    os.replace(file, f"lang2/{EXTENSION_TO_LANGUAGE[ext]}/{os.path.basename(file)}")
