import os

from supported_languages import SUPPORTED_LANGUAGES

if __name__ == '__main__':
    with open("file_names.txt", "w") as file:
        for lang in SUPPORTED_LANGUAGES:
            for filename in os.listdir(f'lang/{lang}'):
                file.write(f'{filename}\n')
