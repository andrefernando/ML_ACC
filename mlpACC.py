import numpy as np
import csv as csv
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold

#Carregando o dataset 
def load_dataset():
    file = "datasetACC/combined_data.data"
    with open(file) as dataset:
        data = np.array(list(csv.reader(dataset)))
        #Removendo linhas duplicadas depois do carregamento do arquivo
        data = np.unique(data,axis=0)
        x_data = np.zeros((len(data)-1,3))
        y_data = np.empty(len(data)-1)

        for x in range(0,len(data)-1):
            x_data[x][0] = data[x][1]
            x_data[x][1] = data[x][2]
            x_data[x][2] = data[x][3]
            y_data[x] = data[x][-1]
    return x_data,y_data

x_data,y_data = load_dataset()

# K-Fold com 10 splits -> número considerado padrão para o mesmo.
kf = KFold(n_splits=10,shuffle=True,random_state=12)

# Parâmetros para a criação da MLP
mlp = MLPClassifier(hidden_layer_sizes=(5,5), max_iter=2000, alpha=1e-4,
                    solver='adam', verbose=1, tol=1e-5, random_state=1,
                    activation = 'logistic',learning_rate_init=.01,shuffle=False)

# Treinamento a rede
for train_indices, test_indices in kf.split(x_data,y_data):
    mlp.fit(x_data[train_indices], y_data[train_indices])
    # Exibindo estatísticas
    print("Training set score: %f" % mlp.score(x_data[train_indices], y_data[train_indices]))
    print("Test set score: %f" % mlp.score(x_data[test_indices], y_data[test_indices]))
    #Capturando os dados das perdas
    loss_values = mlp.loss_curve_

    #Plotando a curva de perdas
    plt.plot(loss_values)
    plt.ylabel("Perda")
    plt.xlabel("Época")
    plt.suptitle('Curva de perda do modelo em treinamento.', size = 16)
    plt.show()

    





