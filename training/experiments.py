import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import LinearSVC

from training.data_splits import collect_features_data, split_train_vali_test
from training.train_models import Trainer


def multinomial_nb_experiment(X_train: np.ndarray, X_vali: np.ndarray, y_train: np.ndarray, y_vali: np.ndarray):
    for alpha in [0, 1, 2, 3, 4, 5]:
        for fit_prior in [True, False]:
            params = {
                "alpha": alpha,
                "fit_prior": fit_prior,
            }
            trainer = Trainer(MultinomialNB, params)
            trainer.train(X_train, y_train)
            print(f"{trainer}: {trainer.score(X_vali, y_vali)}")


def gaussian_nb_experiment(X_train: np.ndarray, X_vali: np.ndarray, y_train: np.ndarray, y_vali: np.ndarray):
    for e in range(20):
        params = {
            "var_smoothing": float(f"1e-{e}"),
        }
        trainer = Trainer(GaussianNB, params)
        trainer.train(X_train, y_train)
        print(f"{trainer}: {trainer.score(X_vali, y_vali)}")


def linear_svc_experiment(X_train: np.ndarray, X_vali: np.ndarray, y_train: np.ndarray, y_vali: np.ndarray):
    experiments = []
    for loss in ["hinge", "squared_hinge"]:
        for penalty in ["l1", "l2"]:
            for e in range(2, 7):
                for c in range(5):
                    for rand in range(3):
                        params = {
                            "loss": loss,
                            "penalty": penalty,
                            "tol": float(f"1e-{e}"),
                            "C": float(c),
                            "random_state": rand,
                        }
                        try:
                            trainer = Trainer(LinearSVC, params)
                            trainer.train(X_train, y_train)
                            score = trainer.score(X_vali, y_vali)
                            experiments.append((trainer, score))
                            print(f"{trainer}: {score}")
                        except ValueError:
                            continue
    print(f"best experiment: {max(experiments, key=lambda tup: tup[1])}")


if __name__ == '__main__':
    data_path = "../data/features_data.jsonl"
    X, y = collect_features_data(data_path, scale=False)
    # X, y = collect_TFIDF_features()

    X_train, X_vali, X_test, y_train, y_vali, y_test = split_train_vali_test(X, y)
    linear_svc_experiment(X_train, X_vali, y_train, y_vali)
