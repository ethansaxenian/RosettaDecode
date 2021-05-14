import math
import pickle
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

import numpy as np
from scipy.stats import mode

from training.data_splits import DataSplitter


class DTree:
    def __init__(self, classes: Optional[List[Any]] = None, criterion: Optional[str] = "gini",
                 max_depth: Optional[int] = None):
        self.classes = classes or [0, 1]
        if criterion not in ("gini", "entropy"):
            raise ValueError('criterion must be in {"gini", "entropy"}')
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = DTreeNode(classes, criterion, max_depth)

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.tree = self.tree.fit(x, y)

    def predict(self, x: List[float]) -> int:
        return self.tree.predict(x)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        total = 0
        accurate = 0

        for sample, label in zip(x, y):
            total += 1
            if self.tree.predict(sample) == label:
                accurate += 1

        return accurate / total

    def save_to_file(self, filename: str):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load_from_file(filename: str) -> 'DTree':
        with open(filename, "rb") as file:
            return pickle.load(file)


class DTreeNode:
    def __init__(self, classes: Optional[List[Any]] = None, criterion: Optional[str] = "gini",
                 max_depth: Optional[int] = None):
        self.classes = classes or [0, 1]
        if criterion not in ("gini", "entropy"):
            raise ValueError('criterion must be in {"gini", "entropy"}')
        self.criterion = criterion
        self.max_depth = max_depth

    def predict(self, x: List[float]) -> int:
        raise NotImplementedError()

    @staticmethod
    def find_candidate_splits(data: List[float]) -> List[float]:
        sorted_points = sorted(set(data))
        midpoints = []
        for i in range(len(sorted_points) - 1):
            midpoints.append((sorted_points[i] + sorted_points[i + 1]) / 2)
        return list(set(midpoints))

    @staticmethod
    def filter(x: np.ndarray, y: np.ndarray, index: int, split: float) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_lt, y_lt, x_gt, y_gt = [], [], [], []
        for sample, label in zip(x, y):
            if sample[index] < split:
                x_lt.append(sample)
                y_lt.append(label)
            else:
                x_gt.append(sample)
                y_gt.append(label)
        return np.array(x_lt), np.array(y_lt), np.array(x_gt), np.array(y_gt)

    def calculate_criterion(self, x: np.ndarray, y: np.ndarray) -> float:
        lang_counts = {lang: 0 for lang in self.classes}
        for label in y:
            lang_counts[label] += 1
        value = 0
        for lang in self.classes:
            p_lang = lang_counts[lang] / len(x)
            if self.criterion == "gini":
                value += p_lang * (1 - p_lang)
            if self.criterion == "entropy":
                value += -p_lang * math.log2(p_lang)
        return value

    def criterion_of_split(self, x: np.ndarray, y: np.ndarray, feature_index: int, split: float) -> float:
        x_lt, y_lt, x_gt, y_gt = self.filter(x, y, feature_index, split)
        return self.calculate_criterion(x_lt, y_lt) + self.calculate_criterion(x_gt, y_gt)

    def fit(self, x: np.ndarray, y: np.ndarray):
        if len(set(y)) == 1:
            return DTreeLeaf(y[0])
        elif self.max_depth == 0:
            return DTreeLeaf(mode(y)[0][0])
        else:
            best_feature, best_split, impurity = 0, 0, 0
            for feature_index in range(len(x[0])):
                feature_set = [sample[feature_index] for sample in x]
                if len(set(feature_set)) == 1:
                    continue
                splits = self.find_candidate_splits(feature_set)
                if best_split == 0:
                    best_feature, best_split = feature_index, splits[0]
                for split_pt in splits:
                    new_impurity = self.criterion_of_split(x, y, feature_index, split_pt)
                    if new_impurity > impurity:
                        best_feature, best_split, impurity = feature_index, split_pt, new_impurity

            x_lt, y_lt, x_gt, y_gt = self.filter(x, y, best_feature, best_split)

            if self.max_depth is not None:
                self.max_depth -= 1
            return DTreeBranch(best_feature, best_split, self.fit(x_lt, y_lt), self.fit(x_gt, y_gt))


@dataclass
class DTreeBranch(DTreeNode):
    def __init__(self, feature: int, split_at: float, less_than: DTreeNode, greater_than: DTreeNode):
        super().__init__()
        self.feature = feature
        self.split_at = split_at
        self.less_than = less_than
        self.greater_than = greater_than

    def predict(self, x: List[float]) -> int:
        if x[self.feature] < self.split_at:
            return self.less_than.predict(x)
        else:
            return self.greater_than.predict(x)


@dataclass
class DTreeLeaf(DTreeNode):
    def __init__(self, estimate: int):
        super().__init__()
        self.estimate = estimate

    def predict(self, x: List[float]) -> int:
        return self.estimate


if __name__ == '__main__':
    splitter = DataSplitter("../data/features/features_data_bc.jsonl", seed=12345)
    X, y = splitter.collect_features_data()
    X_train, X_vali, X_test, y_train, y_vali, y_test = splitter.split_train_vali_test(X, y)

    dtree = DTree(classes=list(range(1, 11)), max_depth=10)

    dtree.fit(X_train, y_train)
    dtree.save_to_file("../data/saved_models/my-dtree")
    # new_tree = DTree.load_from_file("../data/saved_models/my-dtree")
    # print(new_tree.score(X_vali, y_vali))
