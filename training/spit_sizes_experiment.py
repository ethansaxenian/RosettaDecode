from typing import Type, Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample

from shared import RANDOM_SEED, Model
from training.data_splits import DataSplitter
from training.models import MODELS


def test_split_sizes(models: Dict[Type[Model], Dict[str, Any]]):
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
