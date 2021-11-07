import numpy as np
from sklearn.feature_selection import r_regression
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.base import clone
import warnings
warnings.filterwarnings("ignore")
#Zmienne globalne
N_SPLITS = 2
N_REPEATS = 5

FEATURES_RANGE = range(7, 10)

HIDDEN_LAYER_SIZES = [25, 50, 100]
MOMENTUM_VALUES = [0.0, 0.9]

# wyznaczenie rankingu cech za pomocą współczynnika korelacji Pearsona
#feature_ranking = sorted(r_regression(X, y, center=True), reverse=True)
#print(feature_ranking)


def get_data(): # wczytanie zestawu danych
    dataset = np.genfromtxt("wisconsin.csv", delimiter=",")
    X = dataset[1:, :-1]
    y = dataset[1:, -1].astype(int)
    return X, y

def get_classifiers():
    classifiers = {}
    for num_of_features in FEATURES_RANGE:
        for hidden_layer_size in HIDDEN_LAYER_SIZES:
            for momentum_value in MOMENTUM_VALUES:
                new_classifier = mlp(
                    hidden_layer_sizes=(hidden_layer_size,), momentum=momentum_value
                )

                new_classifier.num_of_features = num_of_features

                classifiers[(
                    num_of_features,
                    hidden_layer_size,
                    momentum_value
                )] = new_classifier

    return classifiers

#Ewaluacja z wykorzystaniem protokołu badawczego 5 razy powtórzonej
#2-krotnej walidacji krzyżowej (ang. Cross-validation). Jakość klasyfikacji
#(poprawność diagnozy) należy mierzyć metryką dokładności
#RepeatedKFold repeats K-Fold n times. It can be used when one requires to run KFold n times, producing different splits in each repetition.

def experiment(classifiers, X, y):
    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42
    )
    scores = np.zeros((len(classifiers), N_SPLITS * N_REPEATS))

    for clf_id, clf_name in enumerate(classifiers):
        print(clf_id)
        X_new = SelectKBest(score_func=r_regression, k=classifiers[clf_name].num_of_features).fit_transform(X, y)

        for fold_id, (train, test) in enumerate(rskf.split(X_new, y)):
            clf = clone(classifiers[clf_name])
            print(clf)
            clf.fit(X_new[train], y[train])
            y_pred = clf.predict(X_new[test])
            scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)

    mean = np.mean(scores, axis=1)
    std = np.std(scores, axis=1)

    # for clf_id, clf_name in enumerate(classifiers):
    #     print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))

    #np.save("results", scores)
    np.savetxt("results.csv", scores, delimiter=",")


if __name__ == '__main__':
    X, y = get_data()
    classifiers = get_classifiers()
    experiment(classifiers, X, y)




