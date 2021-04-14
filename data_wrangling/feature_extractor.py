import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import unidecode

from data_wrangling.file_path_storer import generate_file_paths
from globals import get_language_from_filename, DEFAULT_KEYWORDS, SPECIAL_CHAR_NAMES, CHAR_MAPPING, SPECIAL_CHARS


def remove_spaces(code: str) -> str:
    return re.sub(r"[\n\t\s]*", "", code)


def find_words(code: str) -> list[str]:
    return [word for word in re.findall(r'\w+', code) if not word.isdecimal()]


def find_special_characters(code: str) -> list[str]:
    return re.findall(r'[.]{3}|[^0-9a-zA-Z_ \n]', code)  # find all special characters and ellipses


def features_per_line(code: str, regex: Callable[[str], list[str]]) -> list[int]:
    return [len(regex(line)) for line in code.split("\n")]


def n_length_substrings(n: int, code: str) -> list[str]:
    return [code[i:i+n] for i in range(len(code) - (n-1)) if code[i:i+n].isalpha()]


def count_line_endings(code: str) -> Counter[str, int]:
    return Counter(re.findall(r'.(?=\n)', re.sub(r"[ |\t]*", "", code)))


def pct_specials(code: str) -> float:
    return len(find_special_characters(code)) / len(remove_spaces(code))


class FeatureExtractor:
    def __init__(self, path: Optional[str] = None, lowercase: bool = True, binary_counts: bool = False,
                 keywords: Optional[list[str]] = None):
        self.path = path
        self.lowercase = lowercase
        self.binary_counts = binary_counts
        self.reserved_keywords = keywords or DEFAULT_KEYWORDS
        if not Path("../data/file_paths.jsonl").exists():
            generate_file_paths()

    def extract_features(self, code: str) -> dict[str: int]:
        """
        returns a dictionary of features extracted from a string
        """

        features_dict = {}

        bag_of_words = Counter(find_words(code))
        for keyword in self.reserved_keywords:
            num = bag_of_words[keyword]
            features_dict[keyword] = int(bool(num)) if self.binary_counts else num

        specials_count = Counter(find_special_characters(code))
        num_specials = len(list(specials_count.elements()))
        for char in SPECIAL_CHARS:
            features_dict[f'num_{SPECIAL_CHAR_NAMES[char]}'] = int(bool(specials_count[char])) if self.binary_counts else specials_count[char]
            percent_specials = (specials_count[char] / num_specials) if num_specials > 0 else 0
            features_dict[f'percent_{SPECIAL_CHAR_NAMES[char]}'] = percent_specials

        most_common_ending = count_line_endings(code).most_common(1)[0][0]
        features_dict['most_frequent_line_ending'] = CHAR_MAPPING[most_common_ending]

        return features_dict

    def read_file(self, path: str) -> str:
        """
        reads a file path and returns the contents as a string
        """
        with open(path, "r") as file:
            code = file.read()
            if self.lowercase:
                code = code.lower()
            return unidecode.unidecode(code)

    def parse_file(self, path: str) -> dict[str: int]:
        """
        compiles and returns the data from a single file path, which includes the filename, the features dictionary,
        and the language of the code in the file
        """
        code = self.read_file(path)
        return {
            "name": os.path.basename(path),
            "features": self.extract_features(code),
            "lang": get_language_from_filename(path),
            # "code": code,
        }

    def compile_dataset(self):
        """
        stores the features data for each code file in data/features_data.jsonl
        """
        with open(f"../data/{self.path}{'_bc' if self.binary_counts else ''}.jsonl", "w") as outfile, open(
                "../data/file_paths.jsonl", "r") as infile:
            for line in infile:
                info = json.loads(line)
                data = self.parse_file(info["path"])
                json.dump(data, outfile)
                outfile.write("\n")


if __name__ == '__main__':
    extractor = FeatureExtractor("features_data_all", binary_counts=True)
    extractor.compile_dataset()
