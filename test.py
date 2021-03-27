import glob
import os
import pathlib

import unidecode

SUFFIXES = ['.c', '.cpp', '.go', '.hs', '.java', '.js', '.julia', '.pl', '.py', '.rb']


file_list = [f for f in glob.iglob('Go-master/**', recursive=True) if os.path.isfile(f) and pathlib.Path(f).suffix in SUFFIXES]
print(file_list)

for file in file_list:
    os.replace(file, f"go-files/{os.path.basename(file)}")