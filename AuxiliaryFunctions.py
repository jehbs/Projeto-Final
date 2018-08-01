import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt

from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.metrics import silhouette_score


def ApplyPaa(n_paa_segments,seq,df):

    listaFinal = []
    print("segmentos de paa: {}".format(n_paa_segments))
    paa = PiecewiseAggregateApproximation(n_paa_segments)
    scaler = TimeSeriesScalerMeanVariance()
    dadosPaa = pd.DataFrame(seq)
    for i in range(0, len(df)):
        dataset = scaler.fit_transform(df[i])
        paa_dataset_inv = paa.inverse_transform(paa.fit_transform(dataset))
        dadosPaa[i] = paa_dataset_inv[0]
    listaFinal.append(dadosPaa)
    dadosPaa = dadosPaa.T

    plt.plot(dadosPaa)
    plt.show()

    return dadosPaa


def ApplyPca(df,samples):

    pca = PCA(n_components=len(df.columns)).fit(df)
    explained_var = pca.explained_variance_ratio_  # variancia explicada do PCA
    for exp_var_count in range(1, 7):
        sum_var = sum([explained_var[i] for i in range(exp_var_count)])
        print("Variância total dos primeiros {} componentes: {}".format(exp_var_count, sum_var))

    var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100)
    plt.plot(var1)
    plt.show()
    pca = PCA(n_components=20).fit(df)  # aplica a quantidade de componentes prevista pelo teste com as amostras
    reduced_data = pca.fit_transform(df)  # aplicação do pca
    pca_samples = pca.fit_transform(samples)  # idem, ams ans amostras

    reduced_data: DataFrame = pd.DataFrame(reduced_data)

    plt.plot(reduced_data)
    plt.show()
    plt.plot(pca_samples)
    plt.show()

    return(reduced_data,pca_samples)


def SupervisedPreds(df,clf):
    '''
    Aplica um único método de cada vez.
    :param df: dataframe com dados em que se deseja aplicar o aprendizado
    :param clf: classifacador que se deseja usar no aprendizado
    :return: acurácias de treino e teste, f-beta scores de treino e teste e o objeto classificador
    '''

    import warnings
    warnings.filterwarnings("ignore")   #pra não cagar o meu log :D


    classificacao = []
    for i in range(0, int(df.shape[0] / 300)):  # gambiarra para confirmação binária de acerto
        classificacao += [i + 1] * 300

    classi = pd.DataFrame(classificacao)
    X_train, X_test, y_train, y_test = train_test_split(df, classi, test_size=0.3, random_state=0)
    '''
    classifiers = [DecisionTreeClassifier(random_state=20),AdaBoostClassifier(random_state=20),
                   svm.SVC(kernel='linear', C=1, random_state=20),RandomForestClassifier(random_state=20),
                   GaussianNB(),KNeighborsClassifier(),SGDClassifier(random_state=20),
                   LogisticRegression(random_state=20)]
    '''
    print("\nClassificador: {}\n".format(clf.__class__.__name__))
    clf = clf.fit(X_train, y_train)
    clf_test_predictions = clf.predict(X_test)
    clf_train_predictions = clf.predict(X_train)
    acc_train_results = accuracy_score(y_train, clf_train_predictions)
    acc_test_results = accuracy_score(y_test, clf_test_predictions)

    fscore_train_results = fbeta_score(y_train, clf_train_predictions, beta=0.5, average='macro')
    fscore_test_results = fbeta_score(y_test, clf_test_predictions, beta=0.5, average='macro')
    return(acc_train_results,acc_test_results,fscore_train_results,fscore_test_results,clf)


def UnsupervisedPreds(df,samples,clt,components):

    import warnings
    warnings.filterwarnings("ignore")

    for comp in components:
        clt.n_components = comp
        clusterer = clt.fit(df)
        preds = clusterer.predict(df)
        centers = clusterer.means_
        sample_preds = clusterer.predict(samples)  # pca_samples
        score = silhouette_score(df, preds)
    print("score para {} componentes: {}".format(comp, score))

    return (clusterer,preds)
