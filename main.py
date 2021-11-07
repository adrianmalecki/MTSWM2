import numpy as np
import pandas as pd
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

FEATURES_RANGE = range(1, 10)

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
        X_new = SelectKBest(score_func=r_regression, k=classifiers[clf_name].num_of_features).fit_transform(X, y)

        for fold_id, (train, test) in enumerate(rskf.split(X_new, y)):
            clf = clone(classifiers[clf_name])
            print(clf)
            clf.fit(X_new[train], y[train])
            y_pred = clf.predict(X_new[test])
            scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)

    results_of_experiment = {
        'num_of_features': [],
        'hidden_layer_size': [],
        'momentum': [],
        'mean_score': [],
        'std_score': []
    }

    for score, classifier in zip(scores, classifiers):
        mean_score = np.mean(score)
        std_score = np.std(score)
        num_of_features, hidden_layer_size, momentum_value = classifier

        results_of_experiment['num_of_features'].append(num_of_features)
        results_of_experiment['hidden_layer_size'].append(hidden_layer_size)
        results_of_experiment['momentum'].append(momentum_value)
        results_of_experiment['mean_score'].append(mean_score)
        results_of_experiment['std_score'].append(std_score)

    '''
    print("Rezultaty koncowe: (Liczba cech, Liczba neuronow, momentum): Srednia (Odchylenie standardowe)")
    for count in enumerate(results_of_experiment['mean_score']):
        print("MLPClassifier({0}, {1}, {2}): {3:.5f} {4:.4f}".format(results_of_experiment['num_of_features'][count[0]],
                                                              results_of_experiment['hidden_layer_size'][count[0]],
                                                              results_of_experiment['momentum'][count[0]],
                                                              results_of_experiment['mean_score'][count[0]],
                                                              results_of_experiment['std_score'][count[0]]))

    '''
    np.save("results", scores)
    return  results_of_experiment

def show_results(old_results):
    results = pd.DataFrame(old_results)
    print(results.sort_values(by="mean_score", ascending=False))
if __name__ == '__main__':
    X, y = get_data()
    classifiers = get_classifiers()
    result_dict = experiment(classifiers, X, y)
    show_results(result_dict)




