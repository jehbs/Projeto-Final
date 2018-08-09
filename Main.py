# -*- coding: utf-8 -*-
# =-=-=-=-=-=-=-=-
# Projeto de Conclusao de Curso
# Autor: Jéssica Barbosa de Souza
# Descrição : Cógigo Principal. Tem por função gerenciar as leitura, montar os dataFrames,
#             e chamar cada uma das funçoes específicas para realizar o processo previsto
# =-=-=-=-=-=-=-=-

import AuxiliaryFunctions
import pandas as pd
from pandas import DataFrame
import re

import LTSpice_RawRead as LTSpice
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn import svm


if __name__ == "__main__":
    circuitos = ['Sallen Key mc + 4bitPRBS [FALHA].raw', 'Nonlinear Rectfier + 4bit PRBS [FALHA] - 300 - 0.2s.raw',
                'Biquad Highpass Filter mc + 4bitPRBS [FALHA].raw', 'CTSV mc + 4bitPRBS [FALHA].raw']

    #circuitos = ['Biquad Highpass Filter mc + 4bitPRBS [FALHA].raw']

    conjunto = []
    conjunto1 = []
    verificacao = np.zeros((10, 3300))
    dadosReduzidos = []
    dictData = {}
    df1 = pd.DataFrame()
    df = pd.DataFrame()
    dfTime = pd.DataFrame()
    listaFinal, dados = [], []
    n_ts, sz, d = 1, 100, 1

    for circuito in circuitos:
        print("Circuito: {}".format(circuito))
        # =-=-=-=-=-=-=-=-
        # início da leitura do arquivo
        # =-=-=-=-=-=-=-=-
        processa = True
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

        circuito = re.sub('\.', '', circuito)
        fig = plt.figure()
        org = plt.plot(dadosOriginais)
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
        plt.title("Dados pré processados {} ".format(circuito))
        name = "dadosPreProc_{}".format(circuito)
        try:plt.savefig(name, bbox_inches='tight')
        except: plt.savefig(name)

        # =-=-=-=-=-=-=-=-
        # Aplicação do Paa
        # =-=-=-=-=-=-=-=-
        n_paa_segments = 100
        dadosPaa = AuxiliaryFunctions.ApplyPaa(n_paa_segments, matriz, dadosOriginais,circuito)

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # Aplicação do PCA
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        ran = np.random.randint(dadosPaa.shape[0], size=(int(0.1 * dadosPaa.shape[0])))
        samples = pd.DataFrame(dadosPaa.loc[ran], columns=dadosPaa.keys()).reset_index(
            drop=True)  # amostras para treino

        reduced_data, pca_samples = AuxiliaryFunctions.ApplyPca(dadosPaa, samples,circuito)

        fig2 = plt.figure()
        plt.plot(reduced_data.T, '*')
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
        plt.title("Dados pós PCA {} ".format(circuito))
        name = "dadosPosPCA_{}".format(circuito)
        try:plt.savefig(name, bbox_inches='tight')
        except: plt.savefig(name)

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


        classifiers = [DecisionTreeClassifier(random_state=20),AdaBoostClassifier(random_state=20),
                       svm.SVC(kernel='linear', C=1, random_state=20),RandomForestClassifier(random_state=20),
                       GaussianNB(),KNeighborsClassifier(),SGDClassifier(random_state=20),
                       LogisticRegression(random_state=20)]
        k=0
        #classifiers = [GaussianNB()]
        for clf in classifiers:
            acc_train_results, acc_test_results, \
            fscore_train_results, fscore_test_results, \
            clfs = AuxiliaryFunctions.SupervisedPreds(dadosPaa, clf)

            print("acurácia teste: {} \t acurácia treino: {} \n fscore teste: {} \t fscore treino: {} \n".format(
                acc_test_results, acc_train_results, fscore_test_results, fscore_train_results))
            for ct in range(10):
                rd = np.random.randint(0, 3300)
                print("Predição de {}: {}".format(rd, clfs.predict(dadosPaa.iloc[rd, :].values.reshape(1, -1))))

            for j in range(3300):
                verificacao[k][j]= clfs.predict(dadosPaa.iloc[j, :].values.reshape(1, -1))

            fig6 = plt.figure()

            plt.plot(verificacao[k-1].T, '*')
            plt.title("{} para {}".format(clf.__class__.__name__,circuito))

            #fig6.show()
            name = "{}_{}".format(clf.__class__.__name__,circuito,circuito)
            try:plt.savefig(name, bbox_inches='tight')
            except: plt.savefig(name)
            k+=1

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # implementação do modelo de predição não supervisionado
        # modelo: Gaussian Mixed Models
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        from sklearn.mixture import GMM  # importar outro método no lugar do GMM, talvez o dbscan

        range_n_components = list(range(2, 12))
        clusterers = [GMM()]
        for clt in clusterers:
            clts, preds = AuxiliaryFunctions.UnsupervisedPreds(reduced_data, pca_samples, clt, range_n_components)

            for ct in range(10):
                rd = np.random.randint(0, 3300)
                print("Predição de {}: {}".format(rd, clts.predict(reduced_data.iloc[rd, :].values.reshape(1, -1))))

            for j in range(3300):
                verificacao[k][j]= clts.predict(reduced_data.iloc[j, :].values.reshape(1, -1))
            k+=1

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # implementação do modelo de predição não supervisionado
        # modelo: Kmeans
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

        preds, clts = AuxiliaryFunctions.UnsupervisedKmens(reduced_data,pca_samples)

        print("k do kmeans: {}".format(k))
        for ct in range(10):
            rd = np.random.randint(0, 3300)
            print("Predição de {}: {}".format(rd, clts.predict(reduced_data.iloc[rd, :].values.reshape(1, -1))))

        for j in range(3300):
            verificacao[k][j] = clts.predict(reduced_data.iloc[j, :].values.reshape(1, -1))

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # implementação dos teste de validação de resultado
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        moda = [0,0,0,0,0,0,0,0,0,0,0]

        verifica = pd.DataFrame(verificacao)

        for m in range(0,11):
            modLinha = verifica.iloc[k-1]
            mod = modLinha[m*300:299+m*300].mode()[0]
            moda[m] = mod
        print("moda: \n{}".format(moda))

        for n in range(3300):
            #for v in range(0,10,1):
            #    if (modLinha[n]) == moda[v]:
            #        verifica.iloc[k-1][n] = v+1

            if (modLinha[n]) == moda[0]:
                verifica.iloc[k-1][n] = 1
            elif (modLinha[n]) == moda[1]:
                verifica.iloc[k-1][n] = 2
            elif (modLinha[n]) == moda[2]:
                verifica.iloc[k-1][n] = 3
            elif (modLinha[n]) == moda[3]:
                verifica.iloc[k-1][n] = 4
            elif (modLinha[n]) == moda[4]:
                verifica.iloc[k-1][n] = 5
            elif (modLinha[n]) == moda[5]:
                verifica.iloc[k-1][n] = 6
            elif (modLinha[n]) == moda[6]:
                verifica.iloc[k-1][n] = 7
            elif (modLinha[n]) == moda[7]:
                verifica.iloc[k-1][n] = 8
            elif (modLinha[n]) == moda[8]:
                verifica.iloc[k-1][n] = 9
            elif (modLinha[n]) == moda[9]:
                verifica.iloc[k-1][n] = 10
            elif (modLinha[n]) == moda[10]:
                verifica.iloc[k-1][n] = 11

        fig7 = plt.figure()
        plt.plot(verifica.iloc[1].T, '*')
        plt.title("Classificação do KMeans {} ".format(circuito))
        name = "KMeans_{}".format(circuito)
        try:plt.savefig(name, bbox_inches='tight')
        except: plt.savefig(name)

        k = 0
        conjunto = []
        conjunto1 = []
        verificacao = np.zeros((10, 3300))
        dadosReduzidos = []
        dictData = {}
        df1 = pd.DataFrame()
        df = pd.DataFrame()
        dfTime = pd.DataFrame()
        listaFinal, dados = [], []

    print("EOP: pendente relacionar cluster com os componentes do circuito")




