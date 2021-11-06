import numpy as np
from sklearn import clone
from sklearn.feature_selection import r_regression, chi2
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# wczytanie zestawu danych
from sklearn.preprocessing import StandardScaler

dataset = np.genfromtxt("wisconsin.csv", delimiter=",")
X = dataset[1:, :-1]
y = dataset[1:, -1].astype(int)

# wyznaczenie rankingu cech za pomocą współczynnika korelacji Pearsona
feature_ranking = sorted(r_regression(X, y, center=True), reverse=True)
print(feature_ranking)

# ewaluacja

N_SPLITS = 2
N_REPEATS = 5

FEATURES_RANGE = range(1, 8)

HIDDEN_LAYER_SIZES = [20, 50, 90]
MOMENTUM_VALUES = [0.0, 0.9]

def get_classifiers():
    classifiers = {}
    for hidden_layer_size in HIDDEN_LAYER_SIZES:
        for momentum_value in MOMENTUM_VALUES:
            new_classifier = mlp(
                hidden_layer_sizes=(hidden_layer_size,), momentum=momentum_value
            )

            classifiers[(
                hidden_layer_size,
                momentum_value
            )] = new_classifier

    return classifiers


classifiers = get_classifiers()
print(classifiers)

#Ewaluacja z wykorzystaniem protokołu badawczego 5 razy powtórzonej
#2-krotnej walidacji krzyżowej (ang. Cross-validation). Jakość klasyfikacji
#(poprawność diagnozy) należy mierzyć metryką dokładności
#RepeatedKFold repeats K-Fold n times. It can be used when one requires to run KFold n times, producing different splits in each repetition.
rskf = RepeatedStratifiedKFold(
    n_splits=2, n_repeats=5, random_state=4352
)
scores = []
#sc_X = StandardScaler()
for clf in classifiers:
    #print('tu')
    #print(classifiers[clf])
    k = [7,8,9]
    for i in k:
        X_new = SelectKBest(score_func=r_regression, k=i).fit_transform(X, y)
        #print(X_new)
        for train_index, test_index in rskf.split(X_new, y):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            classifiers[clf].fit(X_train, y_train)
            predict = classifiers[clf].predict(X_test)
            scores.append(accuracy_score(y_test, predict))
            print('MLP: ', classifiers[clf], 'k: ', i, 'result: ')
            print(accuracy_score(y_test, predict))
    np.save("results", scores)



