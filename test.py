import glob
import os
import pathlib

import unidecode

SUFFIXES = ['.c', '.cpp', '.go', '.hs', '.java', '.js', '.julia', '.pl', '.py', '.rb']

for lang in ['C-Plus-Plus', 'Haskell', 'Java', 'Javascript', 'Ruby']:
    file_list = [f for f in glob.iglob(f'{lang}-master/**', recursive=True) if os.path.isfile(f) and pathlib.Path(f).suffix in SUFFIXES]
    assert len(file_list) > 0
    os.makedirs(f"{lang}-files", exist_ok=True)

    for file in file_list:
        os.replace(file, f"{lang}-files/{os.path.basename(file)}")