import glob
import os
import pathlib

from globals import LANG_TO_EXT, EXT_TO_LANG

lang = "Julia"
directory = "ModelingToolkit.jl-master/"

file_list = [f for f in glob.iglob(f'{directory}**', recursive=True) if os.path.isfile(f) and pathlib.Path(f).suffix in EXT_TO_LANG.keys()]
assert len(file_list) > 0
# print(file_list)
print(len(file_list))
print(len([f for f in file_list if pathlib.Path(f).suffix == LANG_TO_EXT[lang]]))

# os.makedirs(f"lang2/{lang}", exist_ok=True)
# for file in [f for f in file_list if pathlib.Path(f).suffix == LANG_TO_EXT[lang]]:
#     ext = pathlib.Path(file).suffix
#     os.replace(file, f"lang2/{EXT_TO_LANG[ext]}/{os.path.basename(file)}")

for file in [f for f in file_list if pathlib.Path(f).suffix == LANG_TO_EXT[lang]]:
    ext = pathlib.Path(file).suffix
    os.replace(file, f"lang/{EXT_TO_LANG[ext]}/{os.path.basename(file)}")
