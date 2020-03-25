from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import sklearn.metrics as metrics
import numpy as np
import csv as csv
import matplotlib.pyplot as plt
import os

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
kf = KFold(n_splits=10,shuffle=True,random_state=1)

neigh = KNeighborsClassifier(algorithm='kd_tree', leaf_size=30, metric='minkowski',
            n_jobs=2, n_neighbors=9, p=2, weights='distance')

for train_indices, test_indices in kf.split(x_data,y_data):
    
    neigh.fit(x_data[train_indices], y_data[train_indices]) 
    print("Training set score: %f" % neigh.score(x_data[train_indices], y_data[train_indices]))
    print("Test set score: %f" % neigh.score(x_data[test_indices], y_data[test_indices]))
    print("")
    
    
    

