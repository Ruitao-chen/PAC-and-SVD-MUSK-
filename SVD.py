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

#svd pca
meanValue=np.mean(data1,0)
data1 = data1-meanValue
u, sigma, v = np.linalg.svd(data1[:, :])
svd_pca_new_data = np.dot(u[:,:13],np.diag(sigma)[:13,:13])
#svd_pca_new_data = np.dot(data1,u[:,:13])
print('-------------------------------------svd pca ---------------------------------')
print(sigma[:13])
print(svd_pca_new_data)

df=pd.DataFrame(svd_pca_new_data)
print(df.describe())
df.plot.box(title="SVD")
plt.grid(linestyle="--",alpha=0.3)
plt.show()