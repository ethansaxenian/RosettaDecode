import json
import os
from collections import Counter
from pathlib import Path
from typing import Optional, Dict, List, Any

import unidecode

from data_wrangling.file_path_storer import generate_file_paths
from shared import get_language_from_filename, DEFAULT_KEYWORDS, CHAR_MAPPING, SPECIAL_CHARS, find_words, \
    find_special_characters, count_line_endings


class FeatureExtractor:
    def __init__(self, path: Optional[str] = None, lowercase: bool = True, binary_counts: bool = False,
                 keywords: Optional[List[str]] = None):
        self.path = path
        self.lowercase = lowercase
        self.binary_counts = binary_counts
        self.reserved_keywords = keywords or DEFAULT_KEYWORDS
        if not Path("../data/file_paths.jsonl").exists():
            generate_file_paths()

    def extract_features(self, code: str) -> Dict[str, int]:
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
            num = specials_count[char]
            features_dict[f'{char}'] = int(bool(num)) if self.binary_counts else num
            percent_specials = (specials_count[char] / num_specials) if num_specials > 0 else 0
            features_dict[f'percent_{char}'] = percent_specials

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

    def parse_file(self, path: str) -> Dict[str, Any]:
        """
        compiles and returns the data from a single file path, which includes the filename, the features dictionary,
        and the language of the code in the file
        """
        code = self.read_file(path)
        return {
            "name": os.path.basename(path),
            "features": self.extract_features(code),
            "lang": get_language_from_filename(path),
            "code": code,
        }

    def compile_dataset(self):
        """
        stores the features data for each code file in data/features_data.jsonl
        """
        with open(f"../data/features/{self.path}.jsonl", "w") as outfile, open("../data/file_paths.jsonl", "r") as infile:
            for line in infile:
                info = json.loads(line)
                data = self.parse_file(info["path"])
                json.dump(data, outfile)
                outfile.write("\n")


if __name__ == '__main__':
    extractor = FeatureExtractor("features_data", binary_counts=False)
    extractor.compile_dataset()
