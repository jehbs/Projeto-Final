#Codigo principal
import pandas as pd
import LTSpice_RawRead as LTSpice
import tslearn
import matplotlib.pyplot as plt
import numpy as np
#import visuals as vs
import random

from IPython.display import display
from tslearn.generators import random_walks
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.piecewise import SymbolicAggregateApproximation, OneD_SymbolicAggregateApproximation

from time import time
import logging

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC



if __name__ == "__main__":
    ''' circuitos = ['Sallen Key mc + 4bitPRBS [FALHA].raw', 'Nonlinear Rectfier + 4bit PRBS [FALHA] - 300 - 0.2s.raw',
                 'Biquad Highpass Filter mc + 4bitPRBS [FALHA].raw', 'CTSV mc + 4bitPRBS [FALHA].raw']
                 '''
    circuitos = ['Sallen Key mc + 4bitPRBS [FALHA].raw']
    conjunto = []
    conjunto1 = []

    dadosReduzidos = []
    dictData = {}
    df1 = pd.DataFrame()
    df = pd.DataFrame()
    dfTime = pd.DataFrame()
    n_paa_segments = 30
    listaFinal =[]
    n_components = 1
    n_ts, sz, d = 1, 100, 1
    for circuito in circuitos:
        saida, dados, time = LTSpice.principal(circuito)
        print("leu")
        conjunto.append(saida)
        #print("\nsaída\n")
        #display(saida.axis.data) #tempo
        #display (saida.axis.step_info) #ordinal de simulação
        #display(saida.steps) #ordinal de simulação
        print("\nsaída: tempo x Vout\n")
        #display(saida._traces[10].data)
        #plt.plot(saida._traces[0].data[18559:19486], saida._traces[10].data[18559:19486])

        '''
        saida.traces="nome_do_nó".data
        V(vout) =  saida._traces[10].data
        saida.nPoints/saida.nVariables
        '''
        #print(saida)
        MaiorIndice= 0
        for dado in dados:
            if len(dado)>MaiorIndice:
                MaiorIndice=len(dado)

        print("\n\n")
        matriz =np.zeros((MaiorIndice,len(dados)))

        i=0
        j=0
        for k in range(0, len(saida._traces[10].data)):
            matriz[i][j] = saida._traces[10].data[k]
            if ((saida._traces[10].axis.data[k])==0.0)and (k != 0):
                if ((saida._traces[10].axis.data[k - 1]) != 0.0) :
                    j +=1
                    i=0
                else:
                    i += 1
            else:
                i+=1



        #print(saida)
        dadosOriginais= pd.DataFrame(matriz)
        file_name=circuito+"_original.csv"
        dadosOriginais.to_csv(file_name, sep='\t', encoding='utf-8')
        paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
        scaler = TimeSeriesScalerMeanVariance()
        #dataset1 = scaler.fit_transform(dadosOriginais)
        #paa_dataset_inv1 = paa.inverse_transform(paa.fit_transform(dataset1))
        #plt.title("paa frame")
        #plt.plot(paa_dataset_inv1)
        #plt.show()
        dadosPaa = pd.DataFrame(matriz)
        for i in range(0, len(dados)):
            #plt.plot(time[i], dados[i])
            dataset = scaler.fit_transform(dadosOriginais[i])
            paa_dataset_inv = paa.inverse_transform(paa.fit_transform(dataset))
            #plt.plot(paa_dataset_inv[0])

            dadosPaa[i]=paa_dataset_inv[0]
            #dictData = ({'data':paa_dataset_inv, 'time':time })
            #print("printando o paa inverso")
           # display(paa_dataset_inv)
            #df1time = pd.DataFrame(time)
          #  #df1 = pd.DataFrame(paa_dataset_inv[0].T)
           # df = df.append(df1)
           # dfTime = df.append(df1time)
            #dadosReduzidos.append(paa_dataset_inv)
            #print (paa_dataset_inv)
            #plt.plot(paa_dataset_inv[0].ravel())


        #df = pd.DataFrame(data=dadosReduzidos)

        #plt.title("dados paa")

        #plt.plot(dataset[0].ravel(), "b-", alpha=0.4)
        #plt.plot(paa_dataset_inv[0].ravel(), "b-")
        #plt.show()
        listaFinal.append(dadosPaa)
        plt.figure();
        dadosPaa.plot();


        #X_train, X_test, y_train, y_test = train_test_split(dadosPaa, time, test_size=0.25, random_state=42)
        #print("Extracting the top %d eigenfaces from %d faces"
        #               % (n_components, X_train.shape[0]))
        #n_samples, h, w = dadosPaa.data.shape
        #n_features = dadosPaa.shape[1]
        #n_classes = dadosPaa.shape[0]
        #print("n_features: %d" % n_features)

        #daqui pra baixo trocar "data" por "dadosPaa"
        #ran = random.sample(dadosPaa.shape[1],int(0.1*dadosPaa.shape[1]))
        ran = np.random.randint(dadosPaa.shape[1], size= (int(0.1*dadosPaa.shape[1])))
        #indices = [1, 270, 100] #índices aleatórios de uma conjunto de amostras
        samples = pd.DataFrame(dadosPaa.loc[ran], columns=dadosPaa.keys()).reset_index(drop=True) #amostras ára treino

        # =-=-
        #remoção de "outliers", valores que extrapolam o comportamento do conjunto
        for feature in dadosPaa.keys():
            Q1 = np.percentile(dadosPaa[feature], 5)
            Q3 = np.percentile(dadosPaa[feature], 95)
            step = (Q3 - Q1) * 1.5

            print( "Data points considered outliers for the feature '{}':".format(feature))
            display(dadosPaa[~((dadosPaa[feature] >= Q1 - step) & (dadosPaa[feature] <= Q3 + step))])

        #preencher com os valores identificados como outliers pelo algoritimo e que foram identificados
        #como outliers por vc
        '''outliers = [65, 66, 95, 96, 338, 357,
                    86, 98, 154, 356,
                    75,
                    38, 57, 65, 145, 325, 420,
                    161,
                    109, 138, 167, 142, 184, 187, 203, 233, 285, 289, 343]
        '''
        outliers= []
        #elimina os outliers do dataset
        good_data = dadosPaa.drop(dadosPaa.index[outliers]).reset_index(drop=True)
        # =-=-=-

        #from sklearn.decomposition import PCA

        '''pca = PCA(n_components=n_components, svd_solver='randomized',
                           whiten=True).fit(X_train)
        eigenfaces = pca.components_.reshape((n_components, h, w))'''

        pca = PCA(n_components=len(good_data.columns)).fit(good_data)
        pca_samples = pca.transform(samples)

        explained_var = pca.explained_variance_ratio_ #variancia explicada do PCA
        explained_var1 = sum([explained_var[i] for i in range(1)])  # nas duas primeiras principais
        explained_var2=sum([explained_var[i] for i in range(2)]) #nas duas primeiras principais
        explained_var3 = sum([explained_var[i] for i in range(3)])  # nas duas primeiras principais
        explained_var4=sum([explained_var[i] for i in range(4)]) #nas quatro primeiras
        print ('Variância total dos primeiros 1 componentes:',explained_var1)
        print ('Variância total dos primeiros 2 componentes:',explained_var2)
        print('Variância total dos primeiros 3 componentes:', explained_var3)
        print('Variância total dos primeiros 4 componentes:', explained_var4)
        #pca_results = vs.pca_results(good_data, pca)
       # display(pca_results)


        display(pd.DataFrame(np.round(samples, 4), columns=good_data.index.values))
        pca = PCA(n_components=4).fit(good_data) #aplica a quantidade de componentes prevista pelo teste com as amostras
        # numero de componentes acima deve ser levantado no passo anterior!

        reduced_data = pca.transform(good_data) #aplicação do pca
        pca_samples = pca.transform(samples) #idem

        reduced_data = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2','Dimension 3','Dimension 4'])
        display(reduced_data)

        #X_train_pca = pca.transform(X_train)
        #X_test_pca = pca.transform(X_test)

        #plt.scatter(X_train_pca)
        #plt.scatter(X_test_pca)
        #plt.show()


        #conjunto.append(dadosPaa)
        #display(df)



        print("sucesso")
    df=pd.concat(listaFinal)
    print("sucesso PAA")
    #df.to_csv('teste1.csv', header=None, index=None)

#n_ts, sz, d = 1, 100, 1
#dataset = random_walks(n_ts=n_ts, sz=sz, d=d)
#scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)  # Rescale time series
#dataset = scaler.fit_transform(dataset)

# PAA transform (and inverse transform) of the data


