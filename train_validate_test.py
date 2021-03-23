import json
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

from language_info import LANG_TO_INT

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

    # print(f"Features as {X.shape} matrix.")

    return X, y


def split_train_vali_test(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, train_size=0.75, shuffle=True, random_state=RANDOM_SEED
    )
    X_train, X_vali, y_train, y_vali = train_test_split(
        X_tv, y_tv, train_size=0.66, shuffle=True, random_state=RANDOM_SEED
    )
    # print(X_train.shape, X_vali.shape, X_test.shape)

    return X_train, X_vali, y_train, y_vali


if __name__ == '__main__':
    X, y = parse_features_data("data/features_data.jsonl")

    X_train, X_vali, y_train, y_vali = split_train_vali_test(X, y)
