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
import os

import LTSpice_RawRead as LTSpice
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn import svm


if __name__ == "__main__":
    circuitos = ['Sallen Key mc + 4bitPRBS [FALHA].raw', 'Nonlinear Rectfier + 4bit PRBS [FALHA] - 300 - 0.2s.raw',
                'Biquad Highpass Filter mc + 4bitPRBS [FALHA].raw', 'CTSV mc + 4bitPRBS [FALHA].raw']

    #circuitos = ['Biquad Highpass Filter mc + 4bitPRBS [FALHA].raw']
    #circuitos = ['CTSV mc + 4bitPRBS [FALHA].raw']

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
        csv = re.sub('\.', '', circuito)
        csv = "{}.csv".format(csv)
        # =-=-=-=-=-=-=-=-
        # início da leitura do arquivo
        # =-=-=-=-=-=-=-=-

        if not os.path.isfile(csv):
            print("Obtendo dados do arquivo '{}' .".format(circuito))
            saida, dados, time = LTSpice.principal(circuito)
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
            dadosOriginais.to_csv(csv)
        else:
            print("Obtendo dados do arquivo '{}' .".format(csv))
            dadosOriginais = pd.read_csv(csv, delimiter=";", header=None)
            #conjunto.append(matriz)
            #dadosOriginais = pd.DataFrame(matriz)

        print("Leitura do arquivo terminada.\nSalvando características do circuito...")

        circuito = re.sub('\.', '', circuito)
        circuito = re.sub(' ', '_', circuito)
        fig = plt.figure()
        org = plt.plot(dadosOriginais)
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
        #plt.title("Dados pré processados {} ".format(circuito))
        name = "dadosPreProc_{}".format(circuito)
        try:plt.savefig(name, bbox_inches='tight')
        except: plt.savefig(name)

        # =-=-=-=-=-=-=-=-
        # Aplicação do Paa
        # =-=-=-=-=-=-=-=-
        print("\nIniciando a aplicação do PAA")
        n_paa_segments = 100
        dadosPaa = AuxiliaryFunctions.ApplyPaa(n_paa_segments, matriz, dadosOriginais,circuito)

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # Aplicação do PCA
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        print("\nIniciando a aplicação do PCA")
        ran = np.random.randint(dadosPaa.shape[0], size=(int(0.1 * dadosPaa.shape[0])))
        samples = pd.DataFrame(dadosPaa.loc[ran], columns=dadosPaa.keys()).reset_index(
            drop=True)  # amostras para treino

        reduced_data, pca_samples = AuxiliaryFunctions.ApplyPca(dadosPaa, samples,circuito)

        fig2 = plt.figure()
        plt.plot(reduced_data.T, '*')
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
        #plt.title("Dados pós PCA {} ".format(circuito))
        name = "dadosPosPCA_{}".format(circuito)
        try:plt.savefig(name, bbox_inches='tight')
        except: plt.savefig(name)

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # implementação do modelo de predição supervisionado
        # modelo: 8 modelos diferentes; em destaque: NaiveBayes
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        print("\nIniciando a aplicação dos métodos de aprendizagem supervisionados")
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

            print("Acurácia teste: {}\t Acurácia treino: {}\nFscore teste: {}\t Fscore treino: {}\n".format(
                acc_test_results, acc_train_results, fscore_test_results, fscore_train_results))
            for ct in range(10):
                rd = np.random.randint(0, 3300)
                print("Predição do ponto {}: {}".format(rd, clfs.predict(dadosPaa.iloc[rd, :].values.reshape(1, -1))))

            for j in range(3300):
                verificacao[k][j]= clfs.predict(dadosPaa.iloc[j, :].values.reshape(1, -1))

            fig6 = plt.figure()

            plt.plot(verificacao[k-1].T, '*')
            #plt.title("{} para {}".format(clf.__class__.__name__,circuito))

            #fig6.show()
            name = "{}_{}".format(clf.__class__.__name__,circuito,circuito)
            try:plt.savefig(name, bbox_inches='tight')
            except: plt.savefig(name)
            k+=1

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # implementação do modelo de predição não supervisionado
        # modelo: Gaussian Mixed Models
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        print("\nIniciando a aplicação dos métodos de aprendizagem não-supervisionados")
        from sklearn.mixture import GMM  # importar outro método no lugar do GMM, talvez o dbscan

        range_n_components = list(range(2, 12))
        clusterers = [GMM()]
        for clt in clusterers:
            print("Classificador: {}".format(clt.__class__.__name__))
            clts, preds = AuxiliaryFunctions.UnsupervisedPreds(reduced_data, pca_samples, clt, range_n_components)

            for ct in range(10):
                rd = np.random.randint(0, 3300)
                print("Predição do ponto {}: {}".format(rd, clts.predict(reduced_data.iloc[rd, :].values.reshape(1, -1))))

            for j in range(3300):
                verificacao[k][j]= clts.predict(reduced_data.iloc[j, :].values.reshape(1, -1))
            k+=1

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # implementação do modelo de predição não supervisionado
        # modelo: Kmeans
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        print("Classificador: KMeans")
        preds, clts = AuxiliaryFunctions.UnsupervisedKmeans(reduced_data,pca_samples)

        #print("k do kmeans: {}".format(k))
        for ct in range(10):
            rd = np.random.randint(0, 3300)
            print("Predição de {}: {}".format(rd, clts.predict(reduced_data.iloc[rd, :].values.reshape(1, -1))))

        for j in range(3300):
            verificacao[k][j] = clts.predict(reduced_data.iloc[j, :].values.reshape(1, -1))

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # implementação dos teste de validação de resultado
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        modaKmeans = [0,0,0,0,0,0,0,0,0,0,0]
        modaGMM = [0,0,0,0,0,0,0,0,0,0,0]
        modaLogReg = [0,0,0,0,0,0,0,0,0,0,0]
        modaSGD = [0,0,0,0,0,0,0,0,0,0,0]
        modaKNeigh = [0,0,0,0,0,0,0,0,0,0,0]
        modaNB = [0,0,0,0,0,0,0,0,0,0,0]
        modaRFC = [0,0,0,0,0,0,0,0,0,0,0]
        modaSVC = [0,0,0,0,0,0,0,0,0,0,0]
        modaAda = [0,0,0,0,0,0,0,0,0,0,0]
        modaDTC = [0,0,0,0,0,0,0,0,0,0,0]

        verifica = pd.DataFrame(verificacao)

        linhaKMeans = verifica.iloc[k - 1]
        linhaGMM = verifica.iloc[k - 2]
        linhaLogisticRegression = verifica.iloc[k - 3]
        linhaSGDClassifier = verifica.iloc[k - 4]
        linhaKNeighborsClassifier = verifica.iloc[k - 5]
        linhaGaussianNB = verifica.iloc[k - 6]
        linhaRandomForestClassifier = verifica.iloc[k - 7]
        linhaSVC = verifica.iloc[k - 8]
        linhaAdaBoostClassifier = verifica.iloc[k - 9]
        linhaDecisionTreeClassifier = verifica.iloc[k - 10]

        for m in range(0,11):
            modKmeans = linhaKMeans[m * 300:299 + m * 300].mode()[0]
            modGMM = linhaGMM[m * 300:299 + m * 300].mode()[0]
            modLogReg = linhaLogisticRegression[m * 300:299 + m * 300].mode()[0]
            modSGD = linhaSGDClassifier[m * 300:299 + m * 300].mode()[0]
            modKNeigh = linhaKNeighborsClassifier[m * 300:299 + m * 300].mode()[0]
            modNB = linhaGaussianNB[m * 300:299 + m * 300].mode()[0]
            modRFC = linhaRandomForestClassifier[m * 300:299 + m * 300].mode()[0]
            modSVC = linhaSVC[m * 300:299 + m * 300].mode()[0]
            modAda = linhaAdaBoostClassifier[m * 300:299 + m * 300].mode()[0]
            modDTC = linhaDecisionTreeClassifier[m * 300:299 + m * 300].mode()[0]
            #
            modaKmeans[m] = modKmeans
            modaGMM[m] = modGMM
            modaLogReg[m] = modLogReg
            modaSGD[m] = modSGD
            modaKNeigh[m] = modKNeigh
            modaNB[m] = modNB
            modaRFC[m] = modRFC
            modaSVC[m] = modSVC
            modaAda[m] = modAda
            modaDTC[m] = modDTC

        modas = pd.DataFrame({"modaKmeans":modaKmeans,"modaGMM":modaGMM,"modaLogReg":modaLogReg,
                             "modaSGD":modaSGD,"modaKNeigh":modaKNeigh,"modaNB":modaNB,"modaRFC":modaRFC,
                             "modaSVC":modaSVC,"modaAda":modaAda,"modaDTC":modaDTC})
        print("moda: \n{}".format(modas))

        for n in range(3300):
            #for v in range(0,10,1):
            #    if (modLinha[n]) == moda[v]:
            #        verifica.iloc[k-1][n] = v+1

            if (linhaKMeans[n]) == modaKmeans[0]:
                verifica.iloc[k - 1][n] = 1
            elif (linhaKMeans[n]) == modaKmeans[1]:
                verifica.iloc[k - 1][n] = 2
            elif (linhaKMeans[n]) == modaKmeans[2]:
                verifica.iloc[k - 1][n] = 3
            elif (linhaKMeans[n]) == modaKmeans[3]:
                verifica.iloc[k - 1][n] = 4
            elif (linhaKMeans[n]) == modaKmeans[4]:
                verifica.iloc[k - 1][n] = 5
            elif (linhaKMeans[n]) == modaKmeans[5]:
                verifica.iloc[k - 1][n] = 6
            elif (linhaKMeans[n]) == modaKmeans[6]:
                verifica.iloc[k - 1][n] = 7
            elif (linhaKMeans[n]) == modaKmeans[7]:
                verifica.iloc[k - 1][n] = 8
            elif (linhaKMeans[n]) == modaKmeans[8]:
                verifica.iloc[k - 1][n] = 9
            elif (linhaKMeans[n]) == modaKmeans[9]:
                verifica.iloc[k - 1][n] = 10
            elif (linhaKMeans[n]) == modaKmeans[10]:
                verifica.iloc[k - 1][n] = 11

        fig7 = plt.figure()
        plt.plot(verifica.iloc[k-1].T, '*')
        #plt.title("Classificação do KMeans {} ".format(circuito))
        name = "KMeans_{}".format(circuito)
        try:plt.savefig(name, bbox_inches='tight')
        except: plt.savefig(name)

        fig8 = plt.figure()
        plt.plot(verifica.iloc[k - 2].T, '*')
        #plt.title("Classificação do GMM {} ".format(circuito))
        name = "GMM_{}".format(circuito)
        try:
            plt.savefig(name, bbox_inches='tight')
        except:
            plt.savefig(name)

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




