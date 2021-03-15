"""
Stores all 2-8 length substrings from the data in a corresponding text file
"""
import json

from features import n_length_substrings, find_words
from language_info import LANGUAGE_FILES, get_path_from_filename

if __name__ == '__main__':
    words = set()
    substrings = {
        2: set(),
        3: set(),
        4: set(),
        5: set(),
        6: set(),
        7: set(),
        8: set()
    }
    for filename in LANGUAGE_FILES:
        with open(get_path_from_filename(filename), 'r') as file:
            code = file.read().lower()
            words |= set(find_words(code))
            for i in substrings.keys():
                substrings[i] |= set(n_length_substrings(i, code))

    data = {}
    for n, subs in substrings.items():
        data[n] = {sub: i for i, sub in enumerate(subs)}
    with open(f'data/substrings.jsonl', 'w') as outfile2:
        json.dump(data, outfile2)

    with open(f'data/vocabulary.jsonl', 'w') as outfile1:
        json.dump({word: i for i, word in enumerate(words)}, outfile1)

