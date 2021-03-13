from language_info import LANGUAGE_FILES

if __name__ == '__main__':
    with open("file_names.txt", "w") as file:
        for filename in LANGUAGE_FILES:
            file.write(f'{filename}\n')
