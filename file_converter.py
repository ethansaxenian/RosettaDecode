# This isn't really needed

import os
from shutil import copyfile

from language_info import LANGUAGE_FILES, EXTENSION_TO_LANGUAGE

if __name__ == '__main__':
    os.makedirs('text', exist_ok=True)
    for i, filename in enumerate(LANGUAGE_FILES):
        name, ext = os.path.splitext(filename)
        language = EXTENSION_TO_LANGUAGE[ext]
        try:
            os.mkdir(f'text/{language}')
        except FileExistsError:
            pass
        src = f'lang/{language}/{filename}'
        new_filename = f'{language}-{i + 1}.txt'
        dest = f'text/{language}/{new_filename}'
        copyfile(src, dest)
