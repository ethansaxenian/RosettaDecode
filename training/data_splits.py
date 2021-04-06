import json

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


def collect_TFIDF_features() -> tuple[np.ndarray, np.ndarray]:
    ys = []
    examples = []

    with open("../data/file_paths.jsonl", "r") as file:
        for line in file:
            info = json.loads(line)
            with open(info["path"], "r") as f:
                code = f.read()
                ys.append(LANG_TO_INT[info["lang"]])
                examples.append(code)

    word_to_column = TfidfVectorizer(strip_accents="unicode", lowercase=True, stop_words="english", min_df=100)
    X = word_to_column.fit_transform(examples)
    y = np.array(ys)

    return X.toarray(), y


def collect_features_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    examples = []
    ys = []

    with open(path, "r") as fp:
        for line in fp:
            info = json.loads(line)
            keep = info["features"]
            ys.append(LANG_TO_INT[info["lang"]])
            examples.append(keep)

    feature_numbering = DictVectorizer(sort=True, sparse=False)
    x = feature_numbering.fit_transform(examples)
    X = StandardScaler().fit_transform(x)
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
    # X, y = collect_features_data("../data/features_data.jsonl")
    X, y = collect_TFIDF_features()

    X_train, X_vali, X_test, y_train, y_vali, y_test = split_train_vali_test(X, y)
