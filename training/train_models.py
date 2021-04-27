import numpy as np
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

if __name__ == '__main__':
    # tfidf_vect = TfidfVectorizer(strip_accents="unicode", stop_words="english", min_df=100)
    # count_vectorizer = CountVectorizer(strip_accents="unicode", stop_words="english", min_df=100)

    splitter = DataSplitter("../data/features_data_only_bc.jsonl", seed=RANDOM_SEED, scale=False)
    X, y = splitter.collect_features_data()

    X_train, X_vali, X_test, y_train, y_vali, y_test = splitter.split_train_vali_test(X, y)

    model = MultinomialNB
    params = {
        # 'loss': 'log',
        # 'penalty': 'elasticnet',
        # 'alpha': 0.0001,
        # 'shuffle': False,
        # 'random_state': 1,
        # 'learning_rate': 'constant',
        # 'eta0': 0.01
    }
    trainer = Trainer(model, params, feature_names=splitter.vectorizer.get_feature_names())
    trainer.train(X_train, y_train)
    acc, prec, rec = trainer.validate(X_vali, y_vali)
    print(acc)
    print(prec)
    print(rec)
    # trainer.save_to_file(model.__name__)

    # new_trainer = Trainer.load_from_file("SVC")
    # acc, prec, rec = new_trainer.validate(X_vali, y_vali)
    # print(acc)
    # print(prec)
    # print(rec)

    # for lang, weights_dict in new_trainer.get_feature_weights().items():
    #     print(lang)
    #     for feature, weight in {k: v for k, v in sorted(weights_dict.items(), key=lambda item: item[1], reverse=True)}.items():
    #         print("\t", feature, weight)
    #
    # with open(f"../data/feature_importances_{type(new_trainer.model).__name__}.txt", "w") as file:
    #     for lang, weights_dict in new_trainer.get_feature_weights().items():
    #         file.write(f"{lang}\n")
    #         for feature, weight in {k: v for k, v in sorted(weights_dict.items(), key=lambda item: item[1], reverse=True)}.items():
    #             file.write(f"\tFeature: {feature:<25} Weight: {weight}\n")
