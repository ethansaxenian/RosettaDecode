"""
Creates an excel sheet with labels for each file
"""
import os
import pathlib

from xlwt import Workbook

from data_wrangling.language_info import EXT_TO_LANG, LANG_FILES


def generate_labels():
    os.makedirs("../data", exist_ok=True)
    wb = Workbook()
    labels = wb.add_sheet('labels', cell_overwrite_ok=True)
    labels.write(0, 0, "Filename")
    labels.write(0, 1, "Language")
    labels.write(0, 3, "Supported Languages")

    # list supported languages
    for row, (ext, lang) in enumerate(EXT_TO_LANG.items()):
        labels.write(row + 1, 3, ext)
        labels.write(row + 1, 4, lang)

    # label all supported files
    row = 1
    for filename in LANG_FILES:
        extension = pathlib.Path(filename).suffix
        labels.write(row, 0, filename)
        labels.write(row, 1, EXT_TO_LANG[extension])
        row += 1

    wb.save('../data/labels.xls')


if __name__ == '__main__':
    generate_labels()
