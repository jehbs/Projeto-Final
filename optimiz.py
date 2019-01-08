import sys

import matplotlib


import AuxiliaryFunctions
import pandas as pd
import re
import os

import LTSpice_RawRead as LTSpice
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

print('Python version ' + sys.version)
print('Pandas version ' + pd.__version__)
print('Matplotlib version ' + matplotlib.__version__)
####

conjunto = []
matriz = None

####


#circuito = 'REDUX.raw'
#circuito = 'CTSV mc + 4bitPRBS [FALHA].raw'
#circuitos = ['Sallen Key mc + 4bitPRBS [FALHA].raw', 'Nonlinear Rectfier + 4bit PRBS [FALHA] - 300 - 0.2s.raw',
#               'Biquad Highpass Filter mc + 4bitPRBS [FALHA].raw', 'CTSV mc + 4bitPRBS [FALHA].raw']
#circuitos = ['REDUX.raw']
#circuitos = ['Sallen Key mc + 4bitPRBS [FALHA].raw']
circuitos = ['CTSV mc + 4bitPRBS [FALHA].raw']

for circuito in circuitos:
    print("\n\nCircuito: {}".format(circuito))
    csv_name = re.sub('\.', '', circuito)
    csv_name = "{}.csv".format(csv_name)
    print("\n\nCsv name: {}".format(csv_name))

    dadosOriginais = pd.read_csv(csv_name, header=None, low_memory=False)
    dadosPaa = pd.read_csv("a.csv", header=None, low_memory=False,sep=';')


    from sklearn.model_selection import train_test_split
    from sklearn.metrics import fbeta_score, accuracy_score
    from sklearn.metrics import make_scorer

    classificacao = []
    for i in range(0, int(dadosPaa.shape[0] / 300)):  # gambiarra para confirmação binária de acerto
        classificacao += [i + 1] * 300

    classi = pd.DataFrame(classificacao)
    clf = AdaBoostClassifier(random_state=20)

    # X_train, X_test, y_train, y_test = train_test_split(dadosPaa[:dataSize//10], classi[:dataSize//10], test_size=0.25, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(dadosPaa, classi, test_size=0.25, random_state=0)
    print("Total training subjects: {}\nTotal testing subjects: {}".format(len(X_train), len(X_test)))
    clf = clf.fit(X_train, y_train)
    clf_test_predictions = clf.predict(X_test)
    fscore_test_results = fbeta_score(y_test, clf_test_predictions, beta=0.5, average='macro')


    print("Score Antes da Otimização: {}\n".format(fscore_test_results))

    X_predictions = clf.predict(dadosOriginais.T)

    print("Aplicando tuning do modelo...")
    parameters = {'n_estimators': [200, 400],
                  'learning_rate': [0.1, 0.5, 1.],
                  'base_estimator__min_samples_split': [5, 6, 7, 8],
                  'base_estimator__max_depth': [2, 3, 4, 5]
                  }

    #c,r = y_train.shape
    #y_train = y_train.values.reshape(c,)
    y_train = np.ravel(y_train)

    scorer = make_scorer(fbeta_score, beta=0.5, average='macro')
    # grid_obj = GridSearchCV(clf, parameters, scorer)
    grid_obj = GridSearchCV(clf, parameters, cv=5)
    print(grid_obj)

    grid_fit = grid_obj.fit(X_train, y_train)
    '''
    best_clf = grid_fit.best_estimator_
    predictions = (clf.fit(X_train, y_train)).predict(X_test)
    best_predictions = best_clf.predict(X_test)

    print("Best Score: ", grid_fit.best_score_)
    print("Best Parameters: ", grid_fit.best_params_)
    print("Unoptimized model\n------")
    print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
    print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta=0.5, average='macro')))
    print("\nOptimized Model\n------")
    print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
    print("Final F-score on the testing data: {:.4f}".format(
        fbeta_score(y_test, best_predictions, beta=0.5, average='macro')))

    adjusted_predictions = best_clf.predict(dadosOriginais.T)
    '''