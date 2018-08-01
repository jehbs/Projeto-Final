# Codigo principal
import AuxiliaryFunctions
import pandas as pd
from pandas import DataFrame

import LTSpice_RawRead as LTSpice
import matplotlib.pyplot as plt
import numpy as np



if __name__ == "__main__":
    # circuitos = ['Sallen Key mc + 4bitPRBS [FALHA].raw', 'Nonlinear Rectfier + 4bit PRBS [FALHA] - 300 - 0.2s.raw',
    #            'Biquad Highpass Filter mc + 4bitPRBS [FALHA].raw', 'CTSV mc + 4bitPRBS [FALHA].raw']

    circuitos = ['Nonlinear Rectfier + 4bit PRBS [FALHA] - 300 - 0.2s.raw']
    conjunto = []
    conjunto1 = []


    dadosReduzidos = []
    dictData = {}
    df1 = pd.DataFrame()
    df = pd.DataFrame()
    dfTime = pd.DataFrame()
    listaFinal, dados = [], []
    n_ts, sz, d = 1, 100, 1

    for circuito in circuitos:

        # =-=-=-=-=-=-=-=-
        # início da leitura do arquivo
        # =-=-=-=-=-=-=-=-
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

        # =-=-=-=-=-=-=-=-
        # Aplicação do Paa
        # =-=-=-=-=-=-=-=-
        n_paa_segments = 100
        dadosPaa = AuxiliaryFunctions.ApplyPaa(n_paa_segments, matriz, dadosOriginais)

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # Aplicação do PCA
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        ran = np.random.randint(dadosPaa.shape[0], size=(int(0.1 * dadosPaa.shape[0])))
        samples = pd.DataFrame(dadosPaa.loc[ran], columns=dadosPaa.keys()).reset_index(
            drop=True)  # amostras para treino

        reduced_data, pca_samples = AuxiliaryFunctions.ApplyPca(dadosPaa, samples)
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # implementação do modelo de predição supervisionado
        # modelo: 8 modelos diferentes; em destaque: NaiveBayes
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import SGDClassifier
        from sklearn.linear_model import LogisticRegression

        '''
        classifiers = [DecisionTreeClassifier(random_state=20),AdaBoostClassifier(random_state=20),
                       svm.SVC(kernel='linear', C=1, random_state=20),RandomForestClassifier(random_state=20),
                       GaussianNB(),KNeighborsClassifier(),SGDClassifier(random_state=20),
                       LogisticRegression(random_state=20)]
        '''
        classifiers = [GaussianNB()]
        for clf in classifiers:
            acc_train_results, acc_test_results,\
            fscore_train_results, fscore_test_results,\
            clfs = AuxiliaryFunctions.SupervisedPreds(dadosPaa,clf)

            print("acurácia teste: {} \t acurácia treino: {} \n fscore teste: {} \t fscore treino: {} \n".format(
                acc_test_results, acc_train_results, fscore_test_results, fscore_train_results))
            for ct in range(10):
                rd = np.random.randint(0, 3300)
                print("Predição de {}: {}".format(rd, clfs.predict(dadosPaa.iloc[rd, :].values.reshape(1, -1))))

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # implementação do modelo de predição não supervisionado
        # modelo: Gaussian Mixed Models
        # apropriado para este tipo de dados
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        from sklearn.mixture import GMM  # importar outro método no lugar do GMM, talvez o dbscan

        range_n_components = list(range(2, 12))
        clusterers = [GMM()]
        for clt in clusterers:
            clts, preds = AuxiliaryFunctions.UnsupervisedPreds(reduced_data, pca_samples, clt, range_n_components)
            for i, pred in enumerate(preds):
                print("Sample point", i, "predicted to be in Cluster", pred)

            for ct in range(10):
                rd = np.random.randint(0, 3300)
                print("Predição de {}: {}".format(rd, clts.predict(reduced_data.iloc[rd, :].values.reshape(1, -1))))

    print("EOP: pendente relacionar cluster com os componentes do circuito")
    # df.to_csv('teste1.csv', header=None, index=None)

