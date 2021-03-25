"""
Basic training on the split data with some models we've used in the practicals
Train the models, validate using predict_proba() method and compute accuracy, precision, and recall
"""
from collections import defaultdict
from typing import Type

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from training.split_data import parse_features_data, split_train_vali_test, LANG_TO_INT

INT_TO_LANG = {i: lang for (lang, i) in LANG_TO_INT.items()}

MODELS = {  # note: perceptron has no predict_proba() method
    DecisionTreeClassifier: {
        "criterion": "entropy",
        "max_depth": 8,
        "random_state": 1,
    },
    RandomForestClassifier: {
        "criterion": "entropy",
        "max_depth": 8,
        "random_state": 1,
    },
    LogisticRegression: {
        "random_state": 0,
        "penalty": "l2",
        "max_iter": 3500,
        "C": 1.0,
        "multi_class": "multinomial",
    },
    MLPClassifier: {
        "hidden_layer_sizes": (32, 32, 32, 32),
        "random_state": 0,
        "solver": "adam",
        "learning_rate_init": 0.0001,
        "max_iter": 3500,
        "alpha": 0.0001,
    }
}


def train_model(model: Type[ClassifierMixin], params: dict[str, any], X_train: np.ndarray, y_train: np.ndarray) -> ClassifierMixin:
    f = model(**params)
    print(f"Training {f}.")
    f.fit(X_train, y_train)
    return f


def validate_model(model: ClassifierMixin, X_vali: np.ndarray, y_vali: np.ndarray) -> tuple[float, dict[str, float], dict[str, float]]:
    total = 0
    accurate = 0
    true_pos: dict[str, int] = defaultdict(int)
    false_pos: dict[str, int] = defaultdict(int)
    false_neg: dict[str, int] = defaultdict(int)

    for x, y in zip(X_vali, y_vali):
        probs = model.predict_proba([x])[0]
        pred = list(probs).index(max(probs)) + 1
        # print(probs, "----", y)
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


if __name__ == '__main__':
    data_path = "../data/features_data.jsonl"
    X, y = parse_features_data(data_path)

    X_train, X_vali, X_test, y_train, y_vali, y_test = split_train_vali_test(X, y)

    trained_models = []
    for model, params in MODELS.items():
        trained_models.append(train_model(model, params, X_train, y_train))

    with open("../data/training_data.txt", "a+") as file:
        for model in trained_models:
            acc, prec, rec = validate_model(model, X_vali, y_vali)
            file.write("=====================================================\n")
            file.write(f"stats for {model} with data from {data_path}:\n")
            file.write(f"accuracy: {acc:.3}\n")
            file.write(f"precisions: {dict(sorted(prec.items()))}\n")
            file.write(f"recalls: {dict(sorted(rec.items()))}\n")
