"""
Stores the path of each language file in a text file
"""
import json

from language_info import LANGUAGE_FILES, get_path_from_filename, get_language_from_filename


def generate_file_paths():
    with open("data/file_paths.jsonl", "w") as file:
        for filename in LANGUAGE_FILES:
            data = {
                "path": get_path_from_filename(filename),
                "lang": get_language_from_filename(filename)
            }
            json.dump(data, file)
            file.write("\n")


if __name__ == '__main__':
    generate_file_paths()
