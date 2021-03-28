import glob
import os
import pathlib

EXT_TO_LANG = {
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

LANG_TO_EXT = {l: e for e, l in EXT_TO_LANG.items()}

lang = "C++"
directory = "modern-cpp-tutorial-master/"

file_list = [f for f in glob.iglob(f'{directory}**', recursive=True) if os.path.isfile(f) and pathlib.Path(f).suffix in EXT_TO_LANG.keys()]
assert len(file_list) > 0
print(file_list)
print(len(file_list))
print(len([f for f in file_list if pathlib.Path(f).suffix == LANG_TO_EXT[lang]]))

# os.makedirs(f"lang2/{lang}", exist_ok=True)
#
# for file in [f for f in file_list if pathlib.Path(f).suffix == LANG_TO_EXT[lang]]:
#     ext = pathlib.Path(file).suffix
#     os.replace(file, f"lang2/{EXT_TO_LANG[ext]}/{os.path.basename(file)}")
