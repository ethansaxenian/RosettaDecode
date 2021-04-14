import json
from typing import Union, Optional, Any, Type

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils import resample

from globals import LANG_TO_INT, Model, RANDOM_SEED
from training.models import MODELS


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
            if scale:
                return self.scaler.fit_transform(self.transformer.fit_transform(data))
            return self.scaler.fit_transform(self.transformer.fit_transform(data))
        if scale:
            return self.scaler.transform(self.transformer.transform(data))
        return self.transformer.transform(data)

    def split_train_vali_test(self, X: np.ndarray, y: np.ndarray, scale: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_tv, X_test, y_tv, y_test = train_test_split(
            X, y, train_size=0.75, shuffle=True, random_state=self.random_seed
        )
        X_train, X_vali, y_train, y_vali = train_test_split(
            X_tv, y_tv, train_size=0.66, shuffle=True, random_state=self.random_seed
        )

        return self.prepare_data(X_train, fit=True, scale=scale), self.prepare_data(X_vali, scale=scale), self.prepare_data(X_test, scale=scale), y_train, y_vali, y_test


def test_split_sizes(models: dict[Type[Model], dict[str, Any]]):
    splitter = DataSplitter("../data/features_data_all_bc.jsonl", seed=RANDOM_SEED)
    X, y = splitter.collect_features_data()
    X_train, X_vali, X_test, y_train, y_vali, y_test = splitter.split_train_vali_test(X, y)

    N = len(y_train)
    num_trials = 10
    percentages = list(range(100, 0, -10))

    for model_type, params in models.items():
        scores = {}
        acc_mean = []
        acc_std = []

        print(model_type.__name__)
        for pct in percentages:
            print(f"{pct}% = {int(N * (pct / 100))} samples...")
            scores[pct] = []
            for i in range(num_trials):
                X_sample, y_sample = resample(X_train, y_train, n_samples=int(N * (pct / 100)), replace=False)
                model = model_type(**params)
                if hasattr(model, "random_state"):
                    model.random_state = RANDOM_SEED + pct + i
                model.fit(X_sample, y_sample)
                scores[pct].append(model.score(X_vali, y_vali))
            acc_mean.append(np.mean(scores[pct]))
            acc_std.append(np.std(scores[pct]))

        means = np.array(acc_mean)
        std = np.array(acc_std)
        plt.plot(percentages, acc_mean, "o-")
        plt.fill_between(percentages, means - std, means + std, alpha=0.2, label=model_type.__name__)
    plt.legend()
    plt.xlabel("Percent Train")
    plt.ylabel("Mean Accuracy")
    plt.xlim([0, 100])
    plt.title(f"Shaded Accuracy Plot")
    plt.savefig(f"../data/area-Accuracies.png")
    plt.show()


if __name__ == '__main__':
    test_split_sizes(MODELS)
