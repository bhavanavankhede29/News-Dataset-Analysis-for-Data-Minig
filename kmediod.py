import numpy as np
import scipy.spatial as spt
import numpy.random as rnd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
import seaborn as sns
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
#from scipy.spatial import distance
from Levenshtein import distance
from pyclustering.utils import medoid
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.kmedoids import kmedoids

def squaredEDM(X):
    V=spt.distance.pdist(X,'sqeuclidean')
    #print('V:',V,type(V))
    D=spt.distance.squareform(V)
    #print('D:',D)
    test=spt.distance.sqeuclidean(X[:,0],X[:,1])
    test1=spt.distance.squareform([test])
    #print('test:',test1)
    return D

def kMedoids(D,k,tmax=100):

    #determine dimensions of distance matrix D
    m, n=D.shape

    #randomly initialize an array of k medoid indices
    M=np.sort(np.random.choice(n,k))

    #create a copy of the array of medoid indices
    Mnew=np.copy(M)

    #initalize a dictionary to represent clusters
    C={}

    for t in range(tmax):
        #determine clusters i.e. arrays of data indices
        J=np.argmin(D[:,M],axis=1)

        for kappa in range(k):
            C[kappa]=np.where(J==kappa)[0]

        #update cluster medoids
        for kappa in range(k):
            J=np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            #print('J1:',J,D[np.ix_(C[kappa],C[kappa])],D[np.ix_(C[kappa],C[kappa])].shape)
            try:
                j=np.argmin(J)
                Mnew[kappa]=C[kappa][j]
            except:
                #print('J',J,D[:,M])
                pass
        np.sort(Mnew)

        #check for convergence
        if np.array_equal(M,Mnew):
            break
        M=np.copy(Mnew)
    else:
        #M=np.copy(Mnew)
        #final update of cluster membership
        J=np.argmin(D[:M],axis=1)
        for kappa in range(k):
            C[kappa]=np.where(J==kappa)[0]

    #return results
    return M,C


mean=np.array([0.,0.])
Cov=np.array([[1.,0.],[0.,1.]])
X=rnd.multivariate_normal(mean,Cov,100).T

mu=np.mean(X,axis=1)

D=squaredEDM(X)
j=np.argmin(np.mean(D,axis=1))

me=X[:,j]

plt.scatter(X[0,:],X[1,:],c='green',s=60)
plt.scatter(mu[0],mu[1],marker='s',c='red',s=50)
plt.scatter(me[0],me[1],marker='o',c='blue',s=25)

plt.show()
        
datasetPD=pd.read_csv('articles1.csv',usecols=['publication'])
pubs=['New York Times', 'Atlantic', 'CNN', 'Business Insider','Other']
datasetPD=shuffle(datasetPD)
dataset=[]

for i in range(1000):
    if datasetPD.iloc[i]['publication']==pubs[0]:
        dataset.append(pubs[0])
    elif datasetPD.iloc[i]['publication']==pubs[1]:
        dataset.append(pubs[1])
    elif datasetPD.iloc[i]['publication']==pubs[2]:
        dataset.append(pubs[2])
    elif datasetPD.iloc[i]['publication']==pubs[3]:
        dataset.append(pubs[3])
    else:
        dataset.append(pubs[4])


def computeDistance(numsMatrix):
    Matrix = np.zeros((len(numsMatrix),len(numsMatrix)),dtype=np.int)

    for i in range(0,len(numsMatrix)):
        for j in range(0,len(numsMatrix)):
            Matrix[i,j] = distance(numsMatrix[i],numsMatrix[j])
    return Matrix

distanceMat=computeDistance(dataset)
initial_medoids=[np.random.randint(0,len(distanceMat)) for i in range(5)]


kmedoids_instance = kmedoids(distanceMat, initial_medoids, data_type='distance_matrix')
        # run cluster analysis and obtain results
kmedoids_instance.process()
clusters = kmedoids_instance.get_clusters()
medoids = kmedoids_instance.get_medoids()

for val in clusters:
    print(dataset[val[0]])
    for i in range(1,len(val)):
        print(dataset[val[i]],end='\t')
    print()
    print('*'*100)
    print('*'*100)
    print('*'*100)

barGraphData=[len(val) for val in clusters]
yScale=np.arange(len(medoids))
xScale=[dataset[i] for i in medoids]

plt.bar(yScale, barGraphData, align='center', alpha=0.5)
plt.xticks(yScale, xScale)
plt.ylabel('Usage')
plt.title('Publications')
plt.show()

