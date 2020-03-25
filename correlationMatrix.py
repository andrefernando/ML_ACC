import numpy as np
import csv as csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pandas.plotting import scatter_matrix
import os

#Carregando um dataset 
def load_dataset():
    file = "datasetACC/combined_data.data"
    with open(file) as dataset:
        data = np.array(list(csv.reader(dataset)))
        x_data = np.zeros((len(data)-1,3))
        y_data = np.empty(len(data)-1)

        for x in range(0,len(data)-1):
            x_data[x][0] = data[x][1]
            x_data[x][1] = data[x][2]
            x_data[x][2] = data[x][3]
            y_data[x] = data[x][-1]
    return x_data,y_data

x_data,y_data = load_dataset()

columns = ['frontal','vertical','lateral']
df = pd.DataFrame(x_data,columns=columns)
colMap={1:"red",2:"blue",3:"yellow",4:"green"}
cols=list(map(lambda x:colMap.get(x),y_data))
scatter_matrix(df,diagonal='kde',c=cols,alpha=0.2)
plt.suptitle('Matriz de correlação com dados de aceleração, colorido por atividade.', size = 16)

red_patch = mpatches.Patch(color='red', label='sit on bed')
blue_patch = mpatches.Patch(color='blue', label='sit on chair')
yellow_patch = mpatches.Patch(color='yellow', label='lying')
green_patch = mpatches.Patch(color='green', label='ambulating')
plt.legend(handles=[red_patch,blue_patch,yellow_patch,green_patch])

plt.show()

