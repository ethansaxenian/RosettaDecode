from xlwt import Workbook

from language_info import EXTENSION_TO_LANGUAGE, LANGUAGE_FILES

if __name__ == '__main__':
    wb = Workbook()
    labels = wb.add_sheet('labels', cell_overwrite_ok=True)
    labels.write(0, 0, "Filename")
    labels.write(0, 1, "Language")
    labels.write(0, 3, "Supported Languages")

    # list supported languages
    for row, (ext, lang) in enumerate(EXTENSION_TO_LANGUAGE.items()):
        labels.write(row + 1, 3, ext)
        labels.write(row + 1, 4, lang)

    # label all supported files
    row = 1
    for filename in LANGUAGE_FILES:
        extension = filename[filename.index("."):]
        labels.write(row, 0, filename)
        labels.write(row, 1, EXTENSION_TO_LANGUAGE[extension])
        row += 1

    wb.save('labels.xls')
