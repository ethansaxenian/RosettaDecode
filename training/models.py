from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

MODELS = {
    SGDClassifier: {
        'loss': 'log',
        'penalty': 'elasticnet',
        'alpha': 0.0001,
        'shuffle': False,
        'random_state': 1,
        'learning_rate': 'constant',
        'eta0': 0.01
    },
    LinearSVC: {
        'loss': 'hinge',
        'penalty': 'l2',
        'tol': 0.01,
        'C': 1.0,
        'random_state': 1,
        'max_iter': 20000,
    },
    KNeighborsClassifier: {
        'n_neighbors': 9,
        'weights': 'distance',
        'algorithm': 'auto',
        'leaf_size': 15,
        'p': 1
    },
    MLPClassifier: {
        'hidden_layer_sizes': (100,),
        'activation': 'logistic',
        'solver': 'adam',
        'alpha': 1e-05,
        'batch_size': 64,
        'learning_rate_init': 0.0001,
        'max_iter': 1000,
        'random_state': 1,
        'tol': 0.0001
    },
}
