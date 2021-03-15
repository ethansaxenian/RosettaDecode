"""
Stores the path of each language file in a text file
"""
from language_info import LANGUAGE_FILES, get_path_from_filename

if __name__ == '__main__':
    with open("data/file_paths.txt", "w") as file:
        for filename in LANGUAGE_FILES:
            file.write(f'{get_path_from_filename(filename)}\n')
