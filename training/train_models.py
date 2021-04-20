import pickle
from collections import defaultdict
from typing import Type, Optional

from joblib import dump, load
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from data_wrangling.feature_extractor import FeatureExtractor
from shared import Model, RANDOM_SEED, INT_TO_LANG, DEFAULT_KEYWORDS
from training.data_splits import DataSplitter
from training.models import MODELS


class Trainer:
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

    @staticmethod
    def from_model(model: Model, params: Optional[dict[str, any]] = None, probability: bool = True):
        new_trainer = Trainer(type(model), probability=probability)
        new_trainer.model = model
        if params is not None:
            new_trainer.model.set_params(**new_trainer.params)
        return new_trainer

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        # print(f"Training {self.model}.")
        self.model.fit(X_train, y_train)

    def predict(self, x: np.ndarray) -> np.int64:
        if self.probability and hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba([x])[0]
            print([round(x, 3) for x in probs], end=" ")
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
            if y == pred:
                print(y)
            else:
                print(y, "NOOOOOOOOOOOOOOOOO")

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

    def save_model(self, filename: str):
        with open(f"../data/saved_models/{filename}", "wb") as file:
            pickle.dump(self.model, file)
            print(self.model)

    @staticmethod
    def load_model(filename: str, probability: bool = True):
        with open(f"../data/saved_models/{filename}", "rb") as file:
            model = pickle.load(file)
            return Trainer.from_model(model, probability=probability)


if __name__ == '__main__':
    # tfidf_vect = TfidfVectorizer(strip_accents="unicode", stop_words="english", min_df=100)
    # count_vectorizer = CountVectorizer()
    # tfidf_transform = TfidfTransformer()

    splitter = DataSplitter("../data/features_data_bc.jsonl", seed=RANDOM_SEED)
    X, y = splitter.collect_features_data()

    X_train, X_vali, X_test, y_train, y_vali, y_test = splitter.split_train_vali_test(X, y)

    model_path = "sgd"
    # model = MLPClassifier
    # params = MODELS[model]
    # trainer = Trainer(model, params)
    # trainer.train(X_train, y_train)
    # acc, prec, rec = trainer.validate(X_vali, y_vali)
    # print(acc)
    # print(prec, round(np.mean([i for _, i in prec.items()]), 3))
    # print(rec, round(np.mean([i for _, i in rec.items()]), 3))
    # trainer.save_model(model_path)
    new_trainer = Trainer.load_model(model_path)
    print(new_trainer.predict_sample("../training/train_models.py"))
    # acc, prec, rec = new_trainer.validate(X_vali, y_vali)
    # print(acc)
    # print(prec, round(np.mean([i for _, i in prec.items()]), 3))
    # print(rec, round(np.mean([i for _, i in rec.items()]), 3))
