import pandas as pd
import os
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
sbn.set(color_codes = True)
plt.rcParams['axes.unicode_minus'] = False
from scipy.stats import kstest
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import pyecharts
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


#load data
data1 = pd.read_csv('E:\python\shujuwajue\clean1.data\clean1.data',header = None)
target1 = data1[168].values
y1 = pd.get_dummies(data1[168]).values
data1.drop([1,0,168],axis =1,inplace = True)
data1 = data1.values


#self pca
def pca(x,d):
    meanValue=np.mean(x,0)
    x = x-meanValue
    cov_mat = np.cov(x,rowvar = 0)
    eigVal, eigVec = np.linalg.eig(mat(cov_mat))
    #取最大的d个特征值
    eigValSorted_indices = np.argsort(eigVal)
    eigVec_d = eigVec[:,eigValSorted_indices[:-d-1:-1]] #-d-1前加:才能向左切
    eigVal_d = eigVal[eigValSorted_indices[:-d-1:-1]]
    contributions = round(float(sum(eigVal_d)/sum(eigVal)),2)
    #print("----------------------------eig vectors----------------\n",eigVec_d)
    #print("----------------------------eig values----------------\n",eigVal_d)
    #print("----------------------------contributions----------------\n",contributions)
    return eigVec_d,eigVal_d,contributions

def pca_train(data,max_dim):
    meanValue=np.mean(data,0)
    data = data-meanValue
    for n in range(max_dim):
            eigVec_d,eigVal_d,contributions = pca(data,n)
            if contributions>0.85:
                break
    newdata=np.dot(data,eigVec_d)
    return eigVec_d,eigVal_d,contributions,newdata

#self pca result
print('------------------------------self pca ----------------------------')
eigVec_d,eigVal_d,contributions,newdata = pca_train(data1,max_dim=50)
print("the number of eigVec",len(eigVal_d))
print("eigVec特征值\n",eigVec_d,"\neigVal特征向量\n",eigVal_d,"\ncontributions\n",contributions)
print(newdata)

df=pd.DataFrame(newdata)
print(df.describe())
df.plot.box(title="PAC")
plt.grid(linestyle="--",alpha=0.3)
plt.show()