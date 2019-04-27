import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rd
from sklearn.utils import shuffle
from sklearn.cluster import KMeans

columns=['author','year']
author_key=['Carl Hulse','William McDonald','Choe Sang-Hun']
year_key=[2015,2016,2017]
datasetPD=pd.read_csv('articles1.csv',usecols=columns)
datasetPD=shuffle(datasetPD)
dataset=[]
#print(datasetPD)
authorVal=''
yearVal=''
#Dataset
print('creating dataset....')
for i in range(len(datasetPD)): 
    if datasetPD.iloc[i]['author']==author_key[0]:
        authorVal=rd.uniform(0,5)
    elif datasetPD.iloc[i]['author']==author_key[1]:
        authorVal=rd.uniform(5,10)
    elif datasetPD.iloc[i]['author']==author_key[2]:
        authorVal=rd.uniform(10,15)
    else:
        authorVal=rd.uniform(15,20)

    if datasetPD.iloc[i]['year']==year_key[0]:
        yearVal=rd.uniform(0,5)
    elif datasetPD.iloc[i]['year']==year_key[1]:
        yearVal=rd.uniform(5,10)


    elif datasetPD.iloc[i]['year']==year_key[2]:
        yearVal=rd.uniform(10,15)
    else:
        yearVal=rd.uniform(15,20)

    dataset.append([authorVal,yearVal])
    #print(datasetPD.iloc[i]['author'])
dataset=np.array(dataset)
#points_n=200
clusters_n=4
iteration_n=900


#X=[[1,1],[2,2],[3,3],[4,4],[1,1],[2,2],[1,1],[2,2],[3,3],[4,4],[1,1],[2,2],[3,3],[4,4]]

kMean=KMeans(clusters_n)
kMean.fit(dataset)
clusterCenters=kMean.cluster_centers_
print(clusterCenters)
print('labels:',kMean.labels_)
X=np.array(dataset)
labels=np.array(kMean.labels_)
print(labels)

plt.scatter(X[:,0],X[:,1],c=labels,s=50,alpha=0.5)
plt.plot(clusterCenters[:,0], clusterCenters[:,1],'kx',markersize=15)
plt.show()
