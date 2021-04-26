import json
from typing import Union, Optional

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from shared import LANG_TO_INT


class DataSplitter:
    def __init__(self, path: str, vectorizer: Optional[Union[DictVectorizer, TfidfVectorizer, CountVectorizer]] = None, seed: Optional[int] = None, scale: bool = True):
        self.data_path = path
        self.vectorizer = vectorizer or DictVectorizer(sparse=False)
        self.transformer = TfidfTransformer() if type(self.vectorizer) == CountVectorizer else None
        self.scale = type(self.vectorizer) not in (TfidfVectorizer, CountVectorizer) and scale
        self.scaler = StandardScaler()
        self.random_seed = seed

    def collect_features_data(self) -> tuple[Union[np.ndarray, list[str]], np.ndarray]:
        if type(self.vectorizer) == DictVectorizer:
            features = self._collect_dict_vectorizer_features()

        elif type(self.vectorizer) in (TfidfVectorizer, CountVectorizer):
            features = self._collect_tfidf_features()

        else:
            raise NotImplementedError

        return features

    def _collect_dict_vectorizer_features(self) -> tuple[np.ndarray, np.ndarray]:
        examples = []
        ys = []

        with open(self.data_path, "r") as file:
            for line in file:
                info = json.loads(line)
                examples.append(info["features"])
                ys.append(LANG_TO_INT[info["lang"]])

        return np.array(examples), np.array(ys)

    def _collect_tfidf_features(self) -> tuple[list[str], np.ndarray]:
        examples = []
        ys = []

        with open(self.data_path, "r") as file:
            for line in file:
                info = json.loads(line)
                examples.append(info["code"])
                ys.append(LANG_TO_INT[info["lang"]])

        return examples, np.array(ys)

    def prepare_data(self, data: Union[np.ndarray, list[str]], fit: bool = False) -> np.ndarray:
        if type(self.vectorizer) in (TfidfVectorizer, CountVectorizer):
            assert not self.scale

        if fit:
            if self.scale:
                transformed = self.scaler.fit_transform(self.vectorizer.fit_transform(data))
            else:
                transformed = self.vectorizer.fit_transform(data)
        elif self.scale:
            transformed = self.scaler.transform(self.vectorizer.transform(data))
        else:
            transformed = self.vectorizer.transform(data)

        if type(transformed) != np.ndarray:
            transformed = transformed.toarray()

        return transformed

    def split_train_vali_test(self, X: Union[np.ndarray, list[str]], y: np.ndarray, split_1: float = 0.75, split_2: float = 0.66) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_tv, X_test, y_tv, y_test = train_test_split(X, y, train_size=split_1, random_state=self.random_seed)
        X_train, X_vali, y_train, y_vali = train_test_split(X_tv, y_tv, train_size=split_2, random_state=self.random_seed)

        split_data = (self.prepare_data(X_train, fit=True), self.prepare_data(X_vali), self.prepare_data(X_test), y_train, y_vali, y_test)

        if type(self.vectorizer) == CountVectorizer:
            for split in split_data:
                self.transformer.fit_transform(split.reshape(1, -1))

        return split_data
