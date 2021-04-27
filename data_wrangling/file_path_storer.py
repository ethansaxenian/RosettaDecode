"""
Stores the path of each language file in a text file
"""
import json
import os
from typing import List

from shared import get_path_from_filename, get_language_from_filename, EXT_TO_LANG


def get_lang_files() -> List[str]:
    return [file for lang in EXT_TO_LANG.values() for file in os.listdir(f'../lang/{lang}')]


def generate_file_paths():
    os.makedirs("../data", exist_ok=True)
    with open("../data/file_paths.jsonl", "w") as file:
        for filename in get_lang_files():
            data = {
                "path": get_path_from_filename(filename),
                "lang": get_language_from_filename(filename)
            }
            json.dump(data, file)
            file.write("\n")


if __name__ == '__main__':
    generate_file_paths()
