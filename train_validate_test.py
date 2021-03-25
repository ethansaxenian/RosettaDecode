import json
from dataclasses import dataclass

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from feature_extractor import parse_file

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

# Define & Run Experiments
@dataclass
class ExperimentResult:
    vali_acc: float
    model: ClassifierMixin


def consider_decision_trees():
    print("Consider Decision Tree.")
    performances: list[ExperimentResult] = []

    for rnd in range(3):
        for crit in ["entropy"]:
            for d in range(1, 9):
                params = {
                    "criterion": crit,
                    "max_depth": d,
                    "random_state": rnd,
                }
                f = DecisionTreeClassifier(**params)
                f.fit(X_train, y_train)
                vali_acc = f.score(X_vali, y_vali)
                result = ExperimentResult(vali_acc, f)
                performances.append(result)
    return max(performances, key=lambda result: result.vali_acc)


def consider_random_forest():
    print("Consider Random Forest.")
    performances: list[ExperimentResult] = []
    # Random Forest
    for rnd in range(3):
        for crit in ["entropy"]:
            for d in range(4, 9):
                params = {
                    "criterion": crit,
                    "max_depth": d,
                    "random_state": rnd,
                }
                f = RandomForestClassifier(**params)
                f.fit(X_train, y_train)
                vali_acc = f.score(X_vali, y_vali)
                result = ExperimentResult(vali_acc, f)
                performances.append(result)
    return max(performances, key=lambda result: result.vali_acc)


def consider_perceptron() -> ExperimentResult:
    print("Consider Perceptron.")
    performances: list[ExperimentResult] = []
    for rnd in range(3):
        params = {
            "random_state": rnd,
            "penalty": None,
            "max_iter": 1000,
        }
        f = Perceptron(**params)
        f.fit(X_train, y_train)
        vali_acc = f.score(X_vali, y_vali)
        result = ExperimentResult(vali_acc, f)
        performances.append(result)

    return max(performances, key=lambda result: result.vali_acc)


def consider_logistic_regression() -> ExperimentResult:
    print("Consider Logistic Regression.")
    performances: list[ExperimentResult] = []
    for rnd in range(3):
        params = {
            "random_state": rnd,
            "penalty": "l2",
            "max_iter": 500,
            "C": 1.0,
            "multi_class": "multinomial",
        }
        f = LogisticRegression(**params)
        f.fit(X_train, y_train)
        vali_acc = f.score(X_vali, y_vali)
        result = ExperimentResult(vali_acc, f)
        performances.append(result)

    return max(performances, key=lambda result: result.vali_acc)


def consider_neural_net() -> ExperimentResult:  # Optimized
    print("Consider Multi-Layer Perceptron.")
    performances: list[ExperimentResult] = []
    for rnd in range(3):
        params = {
            "hidden_layer_sizes": (32, 32, 32, 32),
            "random_state": rnd,
            "solver": "adam",
            "learning_rate_init": 0.0001,
            "max_iter": 3500,
            "alpha": 0.0001,
        }
        f = MLPClassifier(**params)
        f.fit(X_train, y_train)
        vali_acc = f.score(X_vali, y_vali)
        result = ExperimentResult(vali_acc, f)
        performances.append(result)

    return max(performances, key=lambda result: result.vali_acc)


if __name__ == '__main__':
    X, y = parse_features_data("data/features_data.jsonl")

    X_train, X_vali, X_test, y_train, y_vali, y_test = split_train_vali_test(X, y)

    # dtree = consider_decision_trees()
    rforest = consider_random_forest()
    # perceptron = consider_perceptron()
    # logit = consider_logistic_regression()
    # mlp = consider_neural_net()

    # print("Best DTree", dtree)
    # print("Best RForest", rforest)
    # print("Best Perceptron", perceptron)
    # print("Best Logistic Regression", logit)
    # print("Best MLP", mlp)

    count = 0
    correct = 0
    for x, y in zip(X_test, y_test):
        pred = rforest.model.predict_proba([x])
        print(pred[0], "----", y)
        if list(pred[0]).index(max(list(pred[0]))) == y - 1:
            correct += 1
        count += 1

    print(correct / count)
