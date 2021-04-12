import json
from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

RANDOM_SEED = 12345678

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


class DataSplitter:
    def __init__(self, path: str, transformer: Union[DictVectorizer, TfidfVectorizer] = DictVectorizer(sparse=False), seed: Optional[int] = None):
        self.data_path = path
        self.transformer = transformer
        self.scaler = StandardScaler()
        self.random_seed = seed

    def collect_features_data(self):
        if type(self.transformer) == DictVectorizer:
            return self._collect_dict_vectorizer_features()

        if type(self.transformer) == TfidfVectorizer:
            return self._collect_tfidf_features()

    def _collect_tfidf_features(self) -> tuple[np.ndarray, np.ndarray]:
        ys = []
        examples = []

        with open("../data/file_paths.jsonl", "r") as file:
            for line in file:
                info = json.loads(line)
                with open(info["path"], "r") as f:
                    code = f.read()
                    ys.append(LANG_TO_INT[info["lang"]])
                    examples.append(code)

        return np.array(examples), np.array(ys)

    def _collect_dict_vectorizer_features(self) -> tuple[np.ndarray, np.ndarray]:
        examples = []
        ys = []

        with open(self.data_path, "r") as fp:
            for line in fp:
                info = json.loads(line)
                examples.append(info["features"])
                ys.append(LANG_TO_INT[info["lang"]])

        return np.array(examples), np.array(ys)

    def prepare_data(self, data: list[dict[str, float]], fit: bool = False, scale: bool = True) -> np.ndarray:
        if fit:
            return self.scaler.fit_transform(self.transformer.fit_transform(data))
        return self.scaler.transform(self.transformer.transform(data))

    def split_train_vali_test(self, X: np.ndarray, y: np.ndarray, scale: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_tv, X_test, y_tv, y_test = train_test_split(
            X, y, train_size=0.75, shuffle=True, random_state=self.random_seed
        )
        X_train, X_vali, y_train, y_vali = train_test_split(
            X_tv, y_tv, train_size=0.66, shuffle=True, random_state=self.random_seed
        )

        return self.prepare_data(X_train, fit=True, scale=scale), self.prepare_data(X_vali, scale=scale), self.prepare_data(X_test, scale=scale), y_train, y_vali, y_test


def test_split_sizes(model_type):
    splitter = DataSplitter("../data/features_data.jsonl", seed=RANDOM_SEED)
    X, y = splitter.collect_features_data()
    X_train, X_vali, X_test, y_train, y_vali, y_test = splitter.split_train_vali_test(X, y)

    N = len(y_train)
    num_trials = 10
    percentages = list(range(100, 0, -10))
    scores = {}
    acc_mean = []
    acc_std = []

    print(N)
    for pct in percentages:
        print(f"{pct}% = {int(N * (pct / 100))} samples...")
        scores[pct] = []
        for i in range(num_trials):
            X_sample, y_sample = resample(X_train, y_train, n_samples=int(N * (pct / 100)), replace=False)
            params = {'hidden_layer_sizes': (100,),
                      'activation': 'logistic',
                      'solver': 'adam',
                      'alpha': 1e-05,
                      'batch_size': 64,
                      'learning_rate_init': 0.0001,
                      'random_state': RANDOM_SEED + pct + i}
            model = model_type(**params)
            model.fit(X_sample, y_sample)
            scores[pct].append(model.score(X_vali, y_vali))
        acc_mean.append(np.mean(scores[pct]))
        acc_std.append(np.std(scores[pct]))

    means = np.array(acc_mean)
    std = np.array(acc_std)
    plt.plot(percentages, acc_mean, "o-")
    plt.fill_between(percentages, means - std, means + std, alpha=0.2)
    plt.xlabel("Percent Train")
    plt.ylabel("Mean Accuracy")
    plt.xlim([0, 100])
    plt.title(f"Shaded Accuracy Plot: {model_type.__name__}")
    plt.savefig(f"../data/area-Accuracy-{model_type.__name__}.png")
    plt.show()


if __name__ == '__main__':
    test_split_sizes(MLPClassifier)
