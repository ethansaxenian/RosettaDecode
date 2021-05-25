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

from shared import RANDOM_SEED, DEFAULT_KEYWORDS
from training.data_splits import DataSplitter
from training.models import MODELS
from training.trainer import Trainer


def save_feature_weights(trainer: Trainer):
    with open(f"../data/feature_importances_{type(trainer.model).__name__}.txt", "w") as file:
        if hasattr(trainer.model, "coef_"):
            for lang, weights_dict in trainer.get_feature_weights().items():
                print(lang)
                file.write(f"{lang}\n")
                for feature, weight in {k: v for k, v in sorted(weights_dict.items(), key=lambda item: item[1], reverse=True)}.items():
                    print("\t", feature, weight)
                    file.write(f"\tFeature: {feature:<25} Weight: {weight}\n")
        elif hasattr(trainer.model, "feature_importances_"):
            for feature, weight in {k: v for k, v in sorted(trainer.get_feature_importances().items(), key=lambda item: item[1], reverse=True)}.items():
                print("\t", feature, weight)
                file.write(f"\tFeature: {feature:<25} Weight: {weight}\n")



if __name__ == '__main__':
    # tfidf_vect = TfidfVectorizer(strip_accents="unicode", stop_words="english", min_df=100)
    # count_vectorizer = CountVectorizer(strip_accents="unicode", stop_words="english", min_df=100)

    # splitter = DataSplitter("../data/features/features_data_bc.jsonl", seed=RANDOM_SEED, scale=False)
    # X, y = splitter.collect_features_data()
    #
    # X_train, X_vali, X_test, y_train, y_vali, y_test = splitter.split_train_vali_test(X, y)

    model = DecisionTreeClassifier
    # params = {
    #     'hidden_layer_sizes': (100,),
    #     'activation': 'logistic',
    #     'solver': 'adam',
    #     'alpha': 1e-05,
    #     'batch_size': 64,
    #     'learning_rate_init': 0.0001,
    #     'max_iter': 1000,
    #     'random_state': 1,
    #     'tol': 0.0001
    # }
    # trainer = Trainer(model, params={}, feature_names=splitter.vectorizer.get_feature_names())
    # trainer.train(X_train, y_train)
    # acc, prec, rec = trainer.validate(X_vali, y_vali)
    # print(acc)
    # print(prec)
    # print(rec)
    #
    # trainer.save_to_file(model.__name__)
    new_trainer = Trainer.load_from_file("DecisionTreeClassifier")

    #
    # acc, prec, rec = new_trainer.validate(X_vali, y_vali)
    # print(acc)
    # print(prec)
    # print(rec)

    save_feature_weights(new_trainer)
