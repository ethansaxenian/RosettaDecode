from collections import defaultdict
from typing import Type, Union, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from data_wrangling.feature_extractor import FeatureExtractor
from training.data_splits import collect_features_data, split_train_vali_test, LANG_TO_INT, collect_TFIDF_features

Model = Union[DecisionTreeClassifier, RandomForestClassifier, LogisticRegression, SGDClassifier, GaussianNB,
              MLPClassifier, SVC, NuSVC, LinearSVC, MultinomialNB]

MODELS = {  # note: perceptron has no predict_proba() method
    # DecisionTreeClassifier: {
    #     "criterion": "entropy",
    #     "max_depth": 8,
    #     "random_state": 1,
    # },
    # RandomForestClassifier: {
    #     "criterion": "entropy",
    #     "max_depth": 8,
    #     "random_state": 1,
    # },
    # LogisticRegression: {
    #     "random_state": 0,
    #     "penalty": "l2",
    #     "max_iter": 3500,
    #     "C": 1.0,
    #     "multi_class": "multinomial",
    # },
    # MLPClassifier: {
    #     "hidden_layer_sizes": (32, 32, 32, 32),
    #     "random_state": 0,
    #     "solver": "adam",
    #     "learning_rate_init": 0.0001,
    #     "max_iter": 3500,
    #     "alpha": 0.0001,
    # },
    # SGDClassifier: {
    #     "loss": "modified_huber",
    #     "random_state": 0,
    # },
    # LinearSVC: {
    #     "random_state": 0,
    #     "max_iter": 1000,
    #     # "dual": False,
    #     "loss": "squared_hinge",
    # },
    # SVC: {  # this takes a WHILE with probibility == True. it also doesn't work very well
    #     # "probability": True,
    #     "random_state": 0,
    # },
    # NuSVC: {  # this takes a WHILE with probibility == True. it also doesn't work very well
    #     # "probability": True,
    #     "random_state": 0,
    # },
    GaussianNB: {},
}


class Trainer:
    int_to_lang = {i: lang for (lang, i) in LANG_TO_INT.items()}

    def __init__(self, model: Type[Model], params: Optional[dict[str, any]] = None, probability: bool = True):
        if params is None:
            params = {}
        if probability and hasattr(model, "probability"):
            params["probability"] = True
        self.params = params
        self.probability = probability
        self.model = model(**self.params)

    def __repr__(self):
        return f"{type(self.model).__name__}{self.params}"

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        # print(f"Training {self.model}.")
        self.model.fit(X_train, y_train)

    def predict(self, x: np.ndarray) -> np.int64:
        if self.probability and hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba([x])[0]
            # print(probs)
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
                true_pos[self.int_to_lang[pred]] += 1
                accurate += 1
            else:
                false_pos[self.int_to_lang[pred]] += 1
                false_neg[self.int_to_lang[y]] += 1

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
        return self.int_to_lang[self.predict(x[0])]


if __name__ == '__main__':
    data_path = "../data/features_data.jsonl"
    X, y = collect_features_data(data_path, scale=False)
    # X, y = collect_TFIDF_features()

    X_train, X_vali, X_test, y_train, y_vali, y_test = split_train_vali_test(X, y)

    # trained_models = []
    # for model, params in MODELS.items():
    #     trainer = Trainer(model, params)
    #     trainer.train(X_train, y_train)
    #     trained_models.append(trainer)
    #
    # for trainer in trained_models:
    #     # print(trainer.model.decision_function(X_vali))
    #     acc, prec, rec = trainer.validate(X_vali, y_vali)
    #     print(f"{acc}\n{prec}\n{rec}")

        # print(trainer.predict_sample("../lang/Go/stack-6.go", binary_counts=True))

    # with open("../data/training_data.txt", "a+") as file:
    #     file.write("=====================================================\n")
    #     file.write(f"data from {data_path}\n")
    #     for trainer in trained_models:
    #         acc, prec, rec = trainer.validate(X_vali, y_vali)
    #         file.write(f"stats for {trainer}:\n")
    #         file.write(f"\taccuracy: {acc:.3}\n")
    #         file.write(f"\tprecisions: {dict(sorted(prec.items()))}\n")
    #         file.write(f"\trecalls: {dict(sorted(rec.items()))}\n\n")
