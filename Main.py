# Codigo principal
import pandas as pd
from pandas import DataFrame

import LTSpice_RawRead as LTSpice
# import tslearn
import matplotlib.pyplot as plt
import numpy as np
# import visuals as vs
# import random
# from IPython import get_ipython
from IPython.display import display
# from tslearn.generators import random_walks
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.piecewise import SymbolicAggregateApproximation, OneD_SymbolicAggregateApproximation

# from time import time
# import logging

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# from tslearn import metrics
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn import svm

if __name__ == "__main__":
    # circuitos = ['Sallen Key mc + 4bitPRBS [FALHA].raw', 'Nonlinear Rectfier + 4bit PRBS [FALHA] - 300 - 0.2s.raw',
    #            'Biquad Highpass Filter mc + 4bitPRBS [FALHA].raw', 'CTSV mc + 4bitPRBS [FALHA].raw']

    circuitos = ['Nonlinear Rectfier + 4bit PRBS [FALHA] - 300 - 0.2s.raw']
    conjunto = []
    conjunto1 = []
    classificacao = []

    dadosReduzidos = []
    dictData = {}
    df1 = pd.DataFrame()
    df = pd.DataFrame()
    dfTime = pd.DataFrame()
    listaFinal, dados = [], []
    n_ts, sz, d = 1, 100, 1

    for circuito in circuitos:


        processa = False
        if processa:
            saida, dados, time = LTSpice.principal(circuito)
            print("leu")
            conjunto.append(saida)
            MaiorIndice = 0
            for dado in dados:
                if len(dado) > MaiorIndice:
                    MaiorIndice = len(dado)

            print("\n\n")
            matriz = np.zeros((MaiorIndice, len(dados)))

            i = 0
            j = 0
            for k in range(0, len(saida._traces[10].data)):
                matriz[i][j] = saida._traces[10].data[k]
                if ((saida._traces[10].axis.data[k]) == 0.0) and (k != 0):
                    if ((saida._traces[10].axis.data[k - 1]) != 0.0):
                        j += 1
                        i = 0
                    else:
                        i += 1
                else:
                    i += 1
            dadosOriginais = pd.DataFrame(matriz)
        else:
            matriz = pd.read_csv('Sallen Key mc + 4bitPRBS [FALHA].csv', delimiter=";", header=None)
            conjunto.append(matriz)
            dadosOriginais = pd.DataFrame(matriz)
            dados = dadosOriginais

        #plt.plot(dados)
        #plt.show()
        #plt.plot(dados.T)
        #plt.show()
        #plt.subplot(4,1,1)
        #plt.plot(dados[150])
        #plt.subplot(4, 1, 2)
        #plt.plot(dados.T[150])
        #plt.show()

        #dadosplt = dadosOriginais.iloc[:,3000:3299]
        #plt.plot(dadosplt)
        #plt.legend()
        #plt.show()

        # =-=-=-=-=-=-=-=-
        # fim da leitura do arquivo
        # =-=-=-=-=-=-=-=-


        # print("escreveu csv")
        # dadosOriginais.to_csv(file_name, sep='\t', encoding='utf-8')
        n_paa_segments = 100
        print("segmentos de paa: {}".format(n_paa_segments))
        paa = PiecewiseAggregateApproximation(n_paa_segments)
        scaler = TimeSeriesScalerMeanVariance()
        dadosPaa = pd.DataFrame(matriz)
        for i in range(0, len(dados)):
            dataset = scaler.fit_transform(dadosOriginais[i])
            paa_dataset_inv = paa.inverse_transform(paa.fit_transform(dataset))
            dadosPaa[i] = paa_dataset_inv[0]
        listaFinal.append(dadosPaa)
        #        plt.figure();
        #        dadosPaa.plot();
        dadosPaa = dadosPaa.T
        # =-=-=-=-=-=-=-=-
        # fim da aplicação do paa
        # =-=-=-=-=-=-=-=-

        # daqui pra baixo trocar "data" por "dadosPaa"
        ran = np.random.randint(dadosPaa.shape[0], size=(int(0.1 * dadosPaa.shape[0])))
        # print("valores dos números escolhidos: {}".format(dadosPaa[ran]))
        samples = pd.DataFrame(dadosPaa.loc[ran], columns=dadosPaa.keys()).reset_index(
            drop=True)  # amostras para treino
        # print( "samples dos números aleatórios escolhidos:  '{}':".format(samples))

        # remoção de "outliers", valores que extrapolam o comportamento do conjunto
        for feature in dadosPaa.keys():
            Q1 = np.percentile(dadosPaa[feature], 1)
            Q3 = np.percentile(dadosPaa[feature], 99)
            step = (Q3 - Q1) * 1.5

            # print( "Data points considered outliers for the feature '{}':".format(feature))
            # display(dadosPaa[~((dadosPaa[feature] >= Q1 - step) & (dadosPaa[feature] <= Q3 + step))])
        '''outliers = [65, 66, 95, 96, 338, 357,
                86, 98, 154, 356,
                75,
                38, 57, 65, 145, 325, 420,
                161,
                109, 138, 167, 142, 184, 187, 203, 233, 285, 289, 343]
        '''
        outliers = [0]
        # elimina os outliers do dataset
        #good_data: object = dadosPaa.drop(dadosPaa.index[outliers]).reset_index(drop=True)
        good_data = dadosPaa
        # good_data = dadosPaa.drop(dadosPaa.index[outliers]).reset_index(drop=True)
        # =-=-=-

        pca = PCA(n_components=len(good_data.columns)).fit(good_data)
        pca_samples = pca.fit_transform(samples)

        explained_var = pca.explained_variance_ratio_  # variancia explicada do PCA
        explained_var1 = sum([explained_var[i] for i in range(1)])  # nas duas primeiras principais
        explained_var2 = sum([explained_var[i] for i in range(2)])  # nas duas primeiras principais
        explained_var3 = sum([explained_var[i] for i in range(3)])  # nas duas primeiras principais
        explained_var4 = sum([explained_var[i] for i in range(4)])  # nas quatro primeiras
        explained_var5 = sum([explained_var[i] for i in range(5)])  # nas quatro primeiras
        explained_var6 = sum([explained_var[i] for i in range(6)])  # nas quatro primeiras
        print('Variância total dos primeiros 1 componentes:', explained_var1)
        print('Variância total dos primeiros 2 componentes:', explained_var2)
        print('Variância total dos primeiros 3 componentes:', explained_var3)
        print('Variância total dos primeiros 4 componentes:', explained_var4)
        print('Variância total dos primeiros 5 componentes:', explained_var5)
        print('Variância total dos primeiros 6 componentes:', explained_var6)

        var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100)
        pca = PCA(n_components=20).fit(good_data)  # aplica a quantidade de componentes prevista pelo teste com as amostras
        reduced_data = pca.fit_transform(good_data)  # aplicação do pca
        pca_samples = pca.fit_transform(samples)  # idem

        reduced_data: DataFrame = pd.DataFrame(reduced_data)
        #display(reduced_data)

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # implementação do modelo de predição supervisionado
        # modelo: SVM
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

        import warnings
        warnings.filterwarnings("ignore")

        from sklearn.metrics import fbeta_score, accuracy_score
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import SGDClassifier
        from sklearn.linear_model import LogisticRegression

        for i in range(0,int(dadosPaa.shape[0] / 300)):
            classificacao += [i + 1] * 300

        classi = pd.DataFrame(classificacao)
        #plt.plot(classi)
        #plt.show()
        X_train, X_test, y_train, y_test = train_test_split(dadosPaa, classi, test_size=0.3, random_state=0)
        '''
        classifiers = [DecisionTreeClassifier(random_state=20),AdaBoostClassifier(random_state=20),
                       svm.SVC(kernel='linear', C=1, random_state=20),RandomForestClassifier(random_state=20),
                       GaussianNB(),KNeighborsClassifier(),SGDClassifier(random_state=20),
                       LogisticRegression(random_state=20)]
        '''
        classifiers = [GaussianNB()]
        for clf in classifiers:
            print("\nClassificador: {}\n".format(clf.__class__.__name__))
            #clf = svm.SVC(kernel='linear', C=1, random_state=20).fit(X_train, y_train)
            clf = clf.fit(X_train, y_train)
            clf_test_predictions = clf.predict(X_test)
            clf_train_predictions = clf.predict(X_train)

            acc_train_results = accuracy_score(y_train, clf_train_predictions)
            acc_test_results = accuracy_score(y_test, clf_test_predictions)

            fscore_train_results = fbeta_score(y_train, clf_train_predictions, beta=0.5, average='macro')
            fscore_test_results = fbeta_score(y_test, clf_test_predictions, beta=0.5, average='macro')

            print("acurácia teste: {} \t acurácia treino: {} \n fscore teste: {} \t fscore treino: {} \n".format(
                acc_test_results,acc_train_results,fscore_test_results,fscore_train_results))
            for ct in range (100):
                rd = np.random.randint(0,3300)
                print("Predição de {}: {}".format(rd,clf.predict(dadosPaa.iloc[rd, :].values.reshape(1, -1))))

        '''
        print("Predição: {}".format(clf.predict(dadosPaa.iloc[1199,:].reshape(1, -1))))

        print("acerto svm: ")
        print(clf.score(X_test, y_test))
        print("acerto svm geral: ")
        print(clf.score(dadosOriginais.T, classi))
        '''

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # implementação do modelo de predição não supervisionado
        # modelo: Gaussian Mixed Models
        # não apropriado para este tipo de dados
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        import warnings

        warnings.filterwarnings("ignore")

        from sklearn.mixture import GMM  # importar outro método no lugar do GMM, talvez o dbscan
        from sklearn.metrics import silhouette_score

        # range_n_components = list(range(2,101))
        # range_n_components = [6]
        range_n_components = list(range(2,12))
        score_comp = []
        for comp in range_n_components:
            clusterer = GMM(n_components=comp).fit(reduced_data)
            preds = clusterer.predict(reduced_data)
            centers = clusterer.means_
            sample_preds = clusterer.predict(pca_samples) #pca_samples
            score = silhouette_score(reduced_data, preds)
            score_comp.append(score)
        print("score para {} componentes: {}".format(comp, score))

        for i, pred in enumerate(preds):
            print("Sample point", i, "predicted to be in Cluster", pred)
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # conjunto.append(dadosPaa)
        # display(df)

        print("sucesso")
    df = pd.concat(listaFinal)
    print("sucesso PAA")
    # df.to_csv('teste1.csv', header=None, index=None)

# n_ts, sz, d = 1, 100, 1
# dataset = random_walks(n_ts=n_ts, sz=sz, d=d)
# scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)  # Rescale time series
# dataset = scaler.fit_transform(dataset)

# PAA transform (and inverse transform) of the data
