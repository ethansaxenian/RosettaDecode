import json
from collections import Counter
from pathlib import Path

from data_wrangling.feature_extractor import FeatureExtractor
from shared import EXT_TO_LANG

if __name__ == '__main__':
    reserved_keywords = set()
    with open("keywords.txt", "r") as file:
        for line in file:
            word = line.strip("\n")
            if word:
                reserved_keywords.add(word.lower())

    # print(sorted(list(reserved_keywords)))

    if not Path("../data/all_keywords.jsonl").exists():
        FeatureExtractor("all_keywords", binary_counts=False, keywords=list(reserved_keywords)).compile_dataset()

    appears = Counter()

    word_frequency = {word: {lang: 0 for lang in EXT_TO_LANG.values()} for word in reserved_keywords}

    with open("../data/all_keywords.jsonl") as f:
        for line in f:
            info = json.loads(line)

            for word in reserved_keywords:
                if info["features"][word]:
                    appears.update([word])
                    word_frequency[word][info["lang"]] += 1

    zero = []
    for word in reserved_keywords:
        if word not in appears:
            zero.append(word)

    # print(appears)
    # print(zero)

    relevant_words = [word for word in appears if appears[word] > 300]
    relevant_words2 = [word for word in appears if max(word_frequency[word].values()) > 200]
    print(len(relevant_words), len(relevant_words2))

    print(set(relevant_words) - set(relevant_words2))
    print(set(relevant_words2) - set(relevant_words))

    with open("../data/keyword_frequencies.txt", "w") as file:
        file.write(f"{'keyword':<12} {''.join(f'{lang:<11}' for lang in EXT_TO_LANG.values())}\n")
        for word in sorted(relevant_words2):
            counts = ''.join([f'{word_frequency[word][lang]:<11}' for lang in EXT_TO_LANG.values()])
            file.write(f"{word + ':':<12} {counts}\n")

    # print(sorted(reserved_keywords))
