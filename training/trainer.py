import pickle
from collections import defaultdict
from typing import Type, Optional, List, Dict

import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import plot_confusion_matrix

from data_wrangling.feature_extractor import FeatureExtractor
from shared import Model, INT_TO_LANG


class Trainer:
    def __init__(self, model: Type[Model], params: Optional[Dict[str, any]] = None, probability: bool = True, feature_names: Optional[List[str]] = None):
        if params is None:
            params = {}
        if probability and hasattr(model(), "probability"):
            params["probability"] = True
        self.params = params
        self.probability = probability
        self.model = model(**self.params)
        self.feature_names = feature_names

    def __repr__(self):
        return f"{type(self.model).__name__}{self.params}"

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        # print(f"Training {self.model}.")
        self.model.fit(X_train, y_train)

    def predict(self, x: np.ndarray) -> np.int64:
        if self.probability and hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba([x])[0]
            # print([round(x, 3) for x in probs], end=" ")
            pred = list(probs).index(max(probs)) + 1
        else:
            pred = self.model.predict([x])[0]

        return pred

    def score(self, X_vali: np.ndarray, y_vali: np.ndarray) -> float:
        return self.model.score(X_vali, y_vali)

    def validate(self, X_vali: np.ndarray, y_vali: np.ndarray) -> tuple[float, dict[str, float], dict[str, float]]:
        # print(f"Validating {self.model}.")
        total = 0
        accurate = 0
        true_pos: dict[str, int] = defaultdict(int)
        false_pos: dict[str, int] = defaultdict(int)
        false_neg: dict[str, int] = defaultdict(int)

        for x, y in zip(X_vali, y_vali):
            pred = self.predict(x)

            total += 1
            if pred == y:
                true_pos[INT_TO_LANG[pred]] += 1
                accurate += 1
            else:
                false_pos[INT_TO_LANG[pred]] += 1
                false_neg[INT_TO_LANG[y]] += 1

        acc = accurate / total

        precisions: dict[str, float] = {}
        for lang in true_pos.keys():
            precisions[lang] = round(true_pos[lang] / (true_pos[lang] + false_pos[lang]), 3)

        recalls: dict[str, float] = {}
        for lang in true_pos.keys():
            recalls[lang] = round(true_pos[lang] / (true_pos[lang] + false_neg[lang]), 3)

        return acc, precisions, recalls

    def predict_sample(self, file_or_code: str, lowercase: bool = True, binary_counts: bool = False):
        feature_extractor = FeatureExtractor(lowercase=lowercase, binary_counts=binary_counts)
        try:
            features = feature_extractor.parse_file(file_or_code)["features"]
        except FileNotFoundError:
            features = feature_extractor.extract_features(file_or_code)

        feature_numbering = DictVectorizer(sparse=False)
        x = feature_numbering.fit_transform([features])
        return INT_TO_LANG[self.predict(x[0])]

    def save_to_file(self, filename: str):
        with open(f"../data/saved_models/{filename}", "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load_from_file(filename: str) -> 'Trainer':
        with open(f"../data/saved_models/{filename}", "rb") as file:
            return pickle.load(file)

    def get_feature_weights(self) -> Dict[str, Dict[str, float]]:
        class_weights = {}
        for class_num, weights in enumerate(self.model.coef_):
            class_weights[INT_TO_LANG[class_num + 1]] = {i: round(v, 3) for i, v in zip(self.feature_names, weights)}
        return class_weights

    def get_feature_importances(self) -> Dict[str, float]:
        return {i: round(v, 3) for i, v in zip(self.feature_names, self.model.feature_importances_)}

    def plot_confusion_matrix(self, X: np.ndarray, y: np.ndarray):
        plot_confusion_matrix(self.model, X, y, normalize="true", display_labels=list(INT_TO_LANG.values()),
                              xticks_rotation="vertical", values_format="0.3f", cmap="Blues")
        plt.show()
