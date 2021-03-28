import json
import sys
from collections import Counter

from data_wrangling.feature_extractor import RESERVED_KEYWORDS
from data_wrangling.language_info import EXT_TO_LANG

if __name__ == '__main__':
    # reserved_keywords = set()
    # with open("words.txt", "r") as file:
    #     for line in file:
    #         word = line.strip("\n")
    #         if word:
    #             reserved_keywords.add(word.lower())

    # print(list(reserved_keywords))

    appears = Counter()

    word_frequency = {lang: {word: 0 for word in RESERVED_KEYWORDS} for lang in EXT_TO_LANG.values()}

    with open("../data/features_data_bc.jsonl") as f:
        for line in f:
            info = json.loads(line)
            counts = info["features"]

            for word in RESERVED_KEYWORDS:
                if counts[word]:
                    appears.update([word])
                    word_frequency[info["lang"]][word] += 1

    zero = []
    for word in RESERVED_KEYWORDS:
        if word not in appears:
            zero.append(word)

    # print(appears)
    # print(zero)

    relevant_words = [word for word in appears if appears[word] > 300]
    print(len(relevant_words))

    final_list = []
    marginal = []

    for word in relevant_words:
        if max([word_frequency[lang][word] for lang in word_frequency.keys()]) < 200:
            marginal.append(word)
        else:
            final_list.append(word)

    for word in final_list:
        print(word, [word_frequency[lang][word] for lang in word_frequency.keys()])

    print(sorted(RESERVED_KEYWORDS))
