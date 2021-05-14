"""

Ethan Saxenian - Project exploration task (5/14) - attempt at implementing a Decision Tree

I started with the sample code you provided and expanded my implementation to include several of the hyperparameters
from sklern's decision tree (criterion, splitter, max_depth, max_features). I think they all work.

I also created the DTree wrapper class for DTreeNode to make things easier for the user - now you don't have to assign
DTreeNode.fit() to a new variable each time. I'm not sure if this was actually necessary, though.

The biggest flaw I've noticed here is that the tree is VERY slow. For this reason I've added the ability to save
the DTree to a file, so you don't have to re-train it each time you want to use it.

I've tried training on my own data set, and am getting around 30% accuracy with no parameter tweaks. This seems pretty
low, which makes me think that I may have done something wrong. However, because it is so slow, I can only train on a
small portion of my data, and larger and larger batches appear to increase performance (but again, this is a very
time-consuming process so I haven't done it that much).

"""
import math
import pickle
import random
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.stats import mode


class DTree:
    """
    wrapper class for DTreeNode
    all attributes attempt to mimic those of sklearn's DecisionTreeClassifier
    """
    def __init__(self, criterion: str = "gini", splitter: str = "best", max_depth: Optional[int] = None,
                 max_features: Optional[Union[int, float, str]] = None, random_state: Optional[int] = None):
        if criterion not in ("gini", "entropy"):
            raise ValueError('criterion must be in {"gini", "entropy"}')
        self.criterion = criterion
        if splitter not in ("best", "random"):
            raise ValueError('splitter must be in {"best", "random"}')
        self.splitter = splitter
        self.max_depth = max_depth
        if isinstance(max_features, float) and not int(max_features) == max_features and max_features > 1.0:
            raise ValueError('max_features must be in (0, n_features]')
        if isinstance(max_features, str) and max_features not in ("sqrt", "log2"):
            raise ValueError('max_features must be in {"sqrt", "log2"}')
        self.max_features = max_features  # N.B.: a small enough value for max_features may result in an error, as all samples may have the same values for these features
        self.random_state = random_state
        self.local_random = random.Random(random_state)

        # this is the actual tree structure that will be trained
        self.tree = DTreeNode(criterion, splitter, max_depth, max_features, random_state)

    def fit(self, x: np.ndarray, y: np.ndarray):
        """ trains tree on data set x, labels y """
        self.classes = sorted(set(y))  # determine class labels from training data
        self.tree.classes = sorted(set(y))
        self.tree = self.tree.fit(x, y)

    def predict(self, x: List[float]) -> int:
        """ classifies list of features x """
        return self.tree.predict(x)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """ returns accuracy of model on data set x, labels y """
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
    def __init__(self, criterion: str = "gini", splitter: str = "best", max_depth: Optional[int] = None,
                 max_features: Optional[Union[int, float, str]] = None, random_state: Optional[int] = None):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.local_random = random.Random(random_state)

    def predict(self, x: List[float]) -> int:
        raise NotImplementedError()

    def find_candidate_splits(self, data: List[float]) -> List[float]:
        """ returns a list of splits to check from data (affected by self.splitter) """
        sorted_points = sorted(set(data))
        midpoints = []
        for i in range(len(sorted_points) - 1):
            midpoints.append((sorted_points[i] + sorted_points[i + 1]) / 2)

        if self.splitter == "best":
            return list(set(midpoints))
        if self.splitter == "random":
            return [self.local_random.choice(list(set(midpoints)))]

    def select_features(self, x: np.ndarray) -> list[int]:
        """ returns the list of features to check splits on (affected by self.max_features) """
        n_features = len(x[0])
        all_features = list(range(n_features))

        if self.max_features is None:
            return all_features
        if isinstance(self.max_features, int):
            return self.local_random.sample(all_features, self.max_features)
        if isinstance(self.max_features, float):
            return self.local_random.sample(all_features, int(self.max_features * n_features))
        if self.max_features == "sqrt":
            return self.local_random.sample(all_features, int(math.sqrt(n_features)))
        if self.max_features == "log2":
            return self.local_random.sample(all_features, int(math.log2(n_features)))

    @staticmethod
    def filter(x: np.ndarray, y: np.ndarray, index: int, split: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ returns a tuple containing the data and labels greater than and less than split at feature_index"""
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
        """ calculates criterion (gini impurity or entropy) for data set x, labels y"""
        lang_counts = {lang: 0 for lang in self.classes}
        for label in y:
            lang_counts[label] += 1

        value = 0
        for lang in self.classes:
            p_lang = lang_counts[lang] / len(x)
            if self.criterion == "gini":
                value += p_lang * (1 - p_lang)
            if self.criterion == "entropy" and p_lang > 0:
                value += -p_lang * math.log2(p_lang)

        return value

    def criterion_of_split(self, x: np.ndarray, y: np.ndarray, feature_index: int, split: float) -> float:
        """ calculates the criterion of data at split for feature_index """
        x_lt, y_lt, x_gt, y_gt = self.filter(x, y, feature_index, split)
        return self.calculate_criterion(x_lt, y_lt) + self.calculate_criterion(x_gt, y_gt)

    def fit(self, x: np.ndarray, y: np.ndarray):
        """ returns a new tree fitted on data set x, labels y """
        if len(set(y)) == 1:  # stop if y has all the same labels
            return DTreeLeaf(y[0])
        elif self.max_depth == 0:  # stop if we've reached max_depth
            return DTreeLeaf(mode(y)[0][0])
        else:
            best_feature, best_split, impurity = 0, 0, 0

            # continue searching over features until a valid partition is found (depends on self.max_features)
            while True:

                for feature_index in self.select_features(x):
                    feature_set = [sample[feature_index] for sample in x]

                    # there won't be any possible splits if all the feature values are the same
                    if len(set(feature_set)) == 1:
                        continue

                    splits = self.find_candidate_splits(feature_set)
                    if best_split == 0:  # default to the first split being the best
                        best_feature, best_split = feature_index, splits[0]

                    for split_pt in splits:
                        new_impurity = self.criterion_of_split(x, y, feature_index, split_pt)
                        if new_impurity < impurity:
                            best_feature, best_split, impurity = feature_index, split_pt, new_impurity

                x_lt, y_lt, x_gt, y_gt = self.filter(x, y, best_feature, best_split)
                if len(x_lt) > 0 and len(x_gt) > 0:
                    break

            if self.max_depth is not None:
                self.max_depth -= 1
            return DTreeBranch(best_feature, best_split, self.fit(x_lt, y_lt), self.fit(x_gt, y_gt))


class DTreeBranch(DTreeNode):
    def __init__(self, feature: int, split_at: float, less_than: DTreeNode, greater_than: DTreeNode):
        # I don't think the various other parameters actually matter here,
        # as all of the logic is coming from the DTreeNode object itself
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


class DTreeLeaf(DTreeNode):
    def __init__(self, estimate: int):
        super().__init__()
        self.estimate = estimate

    def predict(self, x: List[float]) -> int:
        return self.estimate
