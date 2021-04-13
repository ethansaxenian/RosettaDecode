import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from training.data_splits import DataSplitter
from training.train_models import Trainer

RANDOM_SEED = 12345678


def log_experiments(experiments: list[tuple[Trainer, float]]):
    best_model, best_acc = max(experiments, key=lambda tup: tup[1])
    best_experiments = [(model, acc) for model, acc in experiments if acc == best_acc]
    with open("../data/experiments.txt", "a+") as file:
        for model, acc in best_experiments:
            file.write(f"{model} {acc}\n")


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
    log_experiments(experiments)


def linear_svc_experiment_2(X_train: np.ndarray, X_vali: np.ndarray, y_train: np.ndarray, y_vali: np.ndarray):
    experiments = []
    for e in range(2, 10):
        for multi_class in ["ovr", "crammer_singer"]:
            for fit_intercept in [True, False]:
                for rand in range(3):
                    params = {
                        "loss": "squared_hinge",
                        "penalty": "l2",
                        "tol": float(f"1e-{e}"),
                        "C": 1.0,
                        "multi_class": multi_class,
                        "fit_intercept": fit_intercept,
                        "random_state": rand
                    }
                    try:
                        trainer = Trainer(LinearSVC, params)
                        trainer.train(X_train, y_train)
                        score = trainer.score(X_vali, y_vali)
                        experiments.append((trainer, score))
                        print(f"{trainer}: {score}")
                    except ValueError:
                        continue
    log_experiments(experiments)


def linear_svc_experiment_3(X_train: np.ndarray, X_vali: np.ndarray, y_train: np.ndarray, y_vali: np.ndarray):
    experiments = []
    for e in range(2, 10):
        for fit_intercept in [True, False]:
            for rand in range(3):
                params = {
                    "loss": "squared_hinge",
                    "penalty": "l2",
                    "tol": float(f"1e-{e}"),
                    "C": 1.0,
                    "fit_intercept": fit_intercept,
                    "random_state": rand
                }
                try:
                    trainer = Trainer(LinearSVC, params)
                    trainer.train(X_train, y_train)
                    score = trainer.score(X_vali, y_vali)
                    experiments.append((trainer, score))
                    print(f"{trainer}: {score}")
                except ValueError:
                    continue
    log_experiments(experiments)


def sgd_classifier_experiment(X_train: np.ndarray, X_vali: np.ndarray, y_train: np.ndarray, y_vali: np.ndarray):
    experiments = []
    for loss in ["hinge", "squared_hinge", "log", "perceptron"]:
        for penalty in ["l1", "l2", "elasticnet"]:
            for alpha in [0.0001, 0.00001, 0.000001]:
                for shuffle in [True, False]:
                    for rand in range(3):
                        for learning_rate in ["constant", "optimal", "invscaling", "adaptive"]:
                            params = {
                                "loss": loss,
                                "penalty": penalty,
                                "alpha": alpha,
                                "shuffle": shuffle,
                                "random_state": rand,
                                "learning_rate": learning_rate,
                            }
                            try:
                                trainer = Trainer(SGDClassifier, params)
                                trainer.train(X_train, y_train)
                                score = trainer.score(X_vali, y_vali)
                                experiments.append((trainer, score))
                                print(f"{trainer}: {score}")
                            except ValueError:
                                continue
    log_experiments(experiments)


def sgd_classifier_experiment_2(X_train: np.ndarray, X_vali: np.ndarray, y_train: np.ndarray, y_vali: np.ndarray):
    experiments = []
    for rand in range(3):
        for learning_rate in ["constant", "invscaling", "adaptive"]:
            for eta0 in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
                params = {
                    'loss': 'perceptron',
                    'penalty': 'elasticnet',
                    'alpha': 0.0001,
                    'shuffle': True,
                    'random_state': rand,
                    'learning_rate': learning_rate,
                    'eta0': eta0,
                }
                try:
                    trainer = Trainer(SGDClassifier, params)
                    trainer.train(X_train, y_train)
                    score = trainer.score(X_vali, y_vali)
                    experiments.append((trainer, score))
                    print(f"{trainer}: {score}")
                except ValueError:
                    continue
    log_experiments(experiments)


def sgd_classifier_experiment_3(X_train: np.ndarray, X_vali: np.ndarray, y_train: np.ndarray, y_vali: np.ndarray):
    experiments = []
    for loss in ["hinge", "squared_hinge", "log", "perceptron"]:
        for penalty in ["l1", "l2", "elasticnet"]:
            for alpha in [0.0001, 0.00001, 0.000001]:
                for shuffle in [True, False]:
                    for rand in range(3):
                        for learning_rate in ["constant", "invscaling", "adaptive"]:
                            for eta0 in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
                                params = {
                                    "loss": loss,
                                    "penalty": penalty,
                                    "alpha": alpha,
                                    "shuffle": shuffle,
                                    "random_state": rand,
                                    "learning_rate": learning_rate,
                                    "eta0": eta0,
                                }
                                try:
                                    trainer = Trainer(SGDClassifier, params)
                                    trainer.train(X_train, y_train)
                                    score = trainer.score(X_vali, y_vali)
                                    experiments.append((trainer, score))
                                    print(f"{trainer}: {score}")
                                except ValueError:
                                    continue
    log_experiments(experiments)


def k_neighbors_classifier_experiment(X_train: np.ndarray, X_vali: np.ndarray, y_train: np.ndarray, y_vali: np.ndarray):
    experiments = []
    for n_neighbors in range(1, 10):
        for weights in ["uniform", "distance"]:
            for algorithm in ["auto", "ball_tree", "kd_tree", "brute"]:
                for leaf_size in [15, 30, 45]:
                    for p in [1, 2]:
                        params = {
                            "n_neighbors": n_neighbors,
                            "weights": weights,
                            "algorithm": algorithm,
                            "leaf_size": leaf_size,
                            "p": p,
                        }
                        try:
                            trainer = Trainer(KNeighborsClassifier, params)
                            trainer.train(X_train, y_train)
                            score = trainer.score(X_vali, y_vali)
                            experiments.append((trainer, score))
                            print(f"{trainer}: {score}")
                        except ValueError:
                            continue
    log_experiments(experiments)


def decision_tree_experiment(X_train: np.ndarray, X_vali: np.ndarray, y_train: np.ndarray, y_vali: np.ndarray):
    experiments = []
    for criterion in ["gini", "entropy"]:
        for splitter in ["best", "random"]:
            for max_depth in [None, *list(range(1, 21))]:
                for max_features in ["auto", "sqrt", "log2", None]:
                    for rand in range(3):
                        params = {
                            "criterion": criterion,
                            "splitter": splitter,
                            "max_depth": max_depth,
                            "max_features": max_features,
                            "random_state": rand,
                        }
                        try:
                            trainer = Trainer(DecisionTreeClassifier, params)
                            trainer.train(X_train, y_train)
                            score = trainer.score(X_vali, y_vali)
                            experiments.append((trainer, score))
                            print(f"{trainer}: {score}")
                        except ValueError:
                            continue
    log_experiments(experiments)


def mlp_experiment(X_train: np.ndarray, X_vali: np.ndarray, y_train: np.ndarray, y_vali: np.ndarray):
    experiments = []
    for hidden_layer_sizes in range(50, 300, 50):
        for activation in ["identity", "logistic", "tanh", "relu"]:
            for solver in ["lbfgs", "sgd", "adam"]:
                for learning_rate in ["constant", "invscaling", "adaptive"]:
                    for rand in range(3):
                        params = {
                            "hidden_layer_sizes": hidden_layer_sizes,
                            "activation": activation,
                            "solver": solver,
                            "learning_rate": learning_rate,
                            "random_state": rand,
                        }
                        try:
                            trainer = Trainer(MLPClassifier, params)
                            trainer.train(X_train, y_train)
                            score = trainer.score(X_vali, y_vali)
                            experiments.append((trainer, score))
                            print(f"{trainer}: {score}")
                        except ValueError:
                            continue
    log_experiments(experiments)


def mlp_experiment_2(X_train: np.ndarray, X_vali: np.ndarray, y_train: np.ndarray, y_vali: np.ndarray):
    experiments = []
    for alpha in [0.001, 0.0001, 0.00001]:
        for batch_size in [64, 128, 256]:
            for learning_rate_init in [0.01, 0.001, 0.0001]:
                for rand in range(3):
                    params = {
                        "hidden_layer_sizes": (100,),
                        "activation": "logistic",
                        "solver": "adam",
                        "alpha": alpha,
                        "batch_size": batch_size,
                        "learning_rate_init": learning_rate_init,
                        "random_state": rand,
                    }
                    try:
                        trainer = Trainer(MLPClassifier, params)
                        trainer.train(X_train, y_train)
                        score = trainer.score(X_vali, y_vali)
                        experiments.append((trainer, score))
                        print(f"{trainer}: {score}")
                    except ValueError:
                        continue
    log_experiments(experiments)


def mlp_experiment_3(X_train: np.ndarray, X_vali: np.ndarray, y_train: np.ndarray, y_vali: np.ndarray):
    experiments = []
    for batch_size in [50, 64, 100, 128, 200]:
        for rand in range(3):
            params = {
                "hidden_layer_sizes": (100,),
                "activation": "logistic",
                "solver": "adam",
                "alpha": 1e-5,
                "batch_size": batch_size,
                "learning_rate_init": 0.0001,
                "max_iter": 1000,
                "random_state": rand,
                "tol": 1e-4,
            }
            try:
                trainer = Trainer(MLPClassifier, params)
                trainer.train(X_train, y_train)
                score = trainer.score(X_vali, y_vali)
                experiments.append((trainer, score))
                print(f"{trainer}: {score}")
            except ValueError:
                continue
    log_experiments(experiments)


if __name__ == '__main__':
    splitter = DataSplitter("../data/features_data_bc.jsonl", seed=RANDOM_SEED)
    X, y = splitter.collect_features_data()
    X_train, X_vali, X_test, y_train, y_vali, y_test = splitter.split_train_vali_test(X, y)

    # linear_svc_experiment_3(X_train, X_vali, y_train, y_vali)
    # sgd_classifier_experiment(X_train, X_vali, y_train, y_vali)
    # k_neighbors_classifier_experiment(X_train, X_vali, y_train, y_vali)
    # decision_tree_experiment(X_train, X_vali, y_train, y_vali)
    # mlp_experiment_3(X_train, X_vali, y_train, y_vali)
    sgd_classifier_experiment_3(X_train, X_vali, y_train, y_vali)
