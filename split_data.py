"""
Initial train/validate/test setup using essentially the same code from the practicals

features_data.jsonl contains data from 11,848 files in the 10 different languages, fairly evenly distributed.

Currently I have 187 features for each code snippet:
 - presence of 122 reserved keywords from the different languages
 - counts and percentages of each special character
 - the most common line-ending character
Is this too many features?

So far the task seems doable, I'm experimenting with training the various models from p04 and getting 80-90% accuracies.
I would be curious to obtain even more data; currently all my data is from RosettaCode (https://github.com/acmeism/RosettaCodeData).
I have tried downloading the StackOverflow data but ran into issues on my end.
"""
import json

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

LANG_TO_INT = {
    "C": 1,
    "C++": 2,
    "Go": 3,
    "Haskell": 4,
    "Java": 5,
    "JavaScript": 6,
    "Julia": 7,
    "Perl": 8,
    "Python": 9,
    "Ruby": 10
}

RANDOM_SEED = 12345678


def parse_features_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    examples = []
    ys = []

    with open(path, "r") as fp:
        for line in fp:
            info = json.loads(line)
            keep = info["features"]
            ys.append(LANG_TO_INT[info["lang"]])
            examples.append(keep)

    feature_numbering = DictVectorizer(sort=True, sparse=False)
    X = feature_numbering.fit_transform(examples)
    y = np.array(ys)

    return X, y


def split_train_vali_test(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, train_size=0.75, shuffle=True, random_state=RANDOM_SEED
    )
    X_train, X_vali, y_train, y_vali = train_test_split(
        X_tv, y_tv, train_size=0.66, shuffle=True, random_state=RANDOM_SEED
    )

    return X_train, X_vali, X_test, y_train, y_vali, y_test


if __name__ == '__main__':
    X, y = parse_features_data(f"data/features_data.jsonl")
    print(X.shape)

    X_train, X_vali, X_test, y_train, y_vali, y_test = split_train_vali_test(X, y)
