import numpy as np
import pandas as pd
from sklearn.feature_selection import r_regression
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.base import clone
from tabulate import tabulate
from scipy.stats import ttest_rel
import warnings
warnings.filterwarnings("ignore")

#Zmienne globalne
N_SPLITS = 2
N_REPEATS = 5

FEATURES_RANGE = range(9, 10)

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

def upload_scores():
    scores = np.load("results.npy")
    return scores
def t_student(scores):
    alfa = 0.05
    t_statistic = np.zeros((len(classifiers), len(classifiers)))
    p_value = np.zeros((len(classifiers), len(classifiers)))

    for i in range(len(classifiers)):
        for j in range(len(classifiers)):
            t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])

    headers = ["25, 0.0", "25, 0.9", "50, 0.0", "50, 0.9", "100, 0.0", "100, 0.9"]
    names_column = np.array([["25, 0.0"], ["25, 0.9"], ["50, 0.0"], ["50, 0.9"], ["100, 0.0"], ["100, 0.9"]])
    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")

    print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

    advantage = np.zeros((len(classifiers), len(classifiers)))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    print("Advantage:\n", advantage_table)

    significance = np.zeros((len(classifiers), len(classifiers)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    print("Statistical significance (alpha = 0.05):\n", significance_table)

    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)
    print("Statistically significantly better:\n", stat_better_table)


if __name__ == '__main__':
    X, y = get_data()
    classifiers = get_classifiers()
    print("\nExperiment starts\n")
    result_dict = experiment(classifiers, X, y)
    print("\nExperiment ends, results:\n")
    show_results(result_dict)
    print("\nAnalisys starts\n")
    t_student(upload_scores())




