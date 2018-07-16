#Codigo principal
import pandas as pd
import LTSpice_RawRead as LTSpice
import tslearn
import matplotlib.pyplot as plt


from tslearn.generators import random_walks
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.piecewise import SymbolicAggregateApproximation, OneD_SymbolicAggregateApproximation



if __name__ == "__main__":
    circuitos = ['Sallen Key mc + 4bitPRBS [FALHA].raw', 'Nonlinear Rectfier + 4bit PRBS [FALHA] - 300 - 0.2s.raw',
                 'Biquad Highpass Filter mc + 4bitPRBS [FALHA].raw', 'CTSV mc + 4bitPRBS [FALHA].raw']

    conjunto = []
    dadosReduzidos = []
    n_paa_segments = 30
    n_ts, sz, d = 1, 100, 1
    for circuito in circuitos:
        saida, dados, time = LTSpice.principal(circuito)
        conjunto.append(saida)
        #tslearn.piecewise.PiecewiseAggregateApproximation()
        #print(saida)
        paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
        scaler = TimeSeriesScalerMeanVariance()
        for i in range(0, len(dados)):
            #plt.plot(time[i], dados[i])
            dataset = scaler.fit_transform(dados[i])
            paa_dataset_inv = paa.inverse_transform(paa.fit_transform(dataset))
            dadosReduzidos.append(paa_dataset_inv)
            plt.plot(paa_dataset_inv[0].ravel())
        plt.title("dados")
        #plt.plot(dataset[0].ravel(), "b-", alpha=0.4)
        #plt.plot(paa_dataset_inv[0].ravel(), "b-")
        plt.show()

    df = pd.DataFrame(conjunto)
    #df.to_csv('teste1.csv', header=None, index=None)

#n_ts, sz, d = 1, 100, 1
#dataset = random_walks(n_ts=n_ts, sz=sz, d=d)
#scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)  # Rescale time series
#dataset = scaler.fit_transform(dataset)

# PAA transform (and inverse transform) of the data


