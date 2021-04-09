import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

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


def collect_features_data(path: str, scale: bool = True) -> tuple[np.ndarray, np.ndarray]:
    examples = []
    ys = []

    with open(path, "r") as fp:
        for line in fp:
            info = json.loads(line)
            examples.append(info["features"])
            ys.append(LANG_TO_INT[info["lang"]])

    feature_numbering = DictVectorizer(sort=True, sparse=False)
    X = feature_numbering.fit_transform(examples)
    if scale:
        X = StandardScaler().fit_transform(X)
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


def test_split_sizes():
    X, y = collect_features_data("../data/features_data.jsonl")
    X_train, X_vali, X_test, y_train, y_vali, y_test = split_train_vali_test(X, y)

    N = len(y_train)
    num_trials = 100
    sizes = list(range(1000, N, 1000))
    scores = {}
    acc_mean = []
    acc_std = []
    model_type = DecisionTreeClassifier

    print(N)
    for n_samples in sizes:
        print(f"{n_samples} samples...")
        scores[n_samples] = []
        for i in range(num_trials):
            X_sample, y_sample = resample(X_train, y_train, n_samples=n_samples, replace=False)
            model = model_type(random_state=RANDOM_SEED + n_samples + i)
            model.fit(X_sample, y_sample)
            scores[n_samples].append(model.score(X_vali, y_vali))
        acc_mean.append(np.mean(scores[n_samples]))
        acc_std.append(np.std(scores[n_samples]))

    means = np.array(acc_mean)
    std = np.array(acc_std)
    plt.plot(sizes, acc_mean, "o-")
    plt.fill_between(sizes, means - std, means + std, alpha=0.2)
    plt.xlabel("Num Samples")
    plt.ylabel("Mean Accuracy")
    plt.xlim([0, N])
    plt.title("Shaded Accuracy Plot")
    plt.savefig(f"../data/area-Accuracy-{model_type.__name__}.png")
    plt.show()


if __name__ == '__main__':
    # X, y = collect_features_data("../data/features_data.jsonl")
    # X, y = collect_TFIDF_features()

    # X_train, X_vali, X_test, y_train, y_vali, y_test = split_train_vali_test(X, y)
    test_split_sizes()
