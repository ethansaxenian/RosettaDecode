"""
Stores the path of each language file in a text file
"""
import json
import os
import pathlib

from data_wrangling.language_info import EXT_TO_LANG, get_language_from_filename, LANG_FILES


def get_path_from_filename(filename: str) -> str:
    ext = pathlib.Path(filename).suffix
    language = EXT_TO_LANG[ext]
    return f'../lang/{language}/{filename}'


def generate_file_paths():
    os.makedirs("../data", exist_ok=True)
    with open("../data/file_paths.jsonl", "w") as file:
        for filename in LANG_FILES:
            data = {
                "path": get_path_from_filename(filename),
                "lang": get_language_from_filename(filename)
            }
            json.dump(data, file)
            file.write("\n")


if __name__ == '__main__':
    generate_file_paths()
