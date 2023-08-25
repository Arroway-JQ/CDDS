#%%
import os
import sys
sys.path.append("/home3/")
import numpy as np
import matplotlib.pyplot as plt
from skfuzzy.cluster import cmeans_predict
import scipy.io as sio
from multiprocessing import Pool as ThreadPool
import joblib
import h5py
import math
print('Import lib done')

#%%
# prepare data

analyDir = '/home3/Dspectraanaly/'
matfn = analyDir + 'intestDspectraSWT_all.mat'
data = h5py.File(matfn)
type(data)
testDspectraSWT = data['intestDspectraSWT'][:]

#testDspectraWT = testDspectraWT.T
#np.array(trainDspectraWT, dtype = 'float32')
np.mat(testDspectraSWT)
#trainDspectraWT = trainDspectraWT.T    #FCM模型中数据的输入需要转置一下
dim = testDspectraSWT.shape
print(dim)
m_t = 1 + (1418/dim[1]+22.05)*(dim[0]**(-2))+(12.33/dim[1]+0.243)*(dim[0]**(-0.0406*math.log(dim[1])-0.1134))

#m_t = 1.026296483456355
#matfn2 = analyDir + 'fcmresults2/FCM03cluster_1.0317349776880484_results.mat'
#data2 =  sio.loadmat(matfn2)
ncenters = data['centers'][:]
data.close()
np.array(ncenters, dtype = 'float32')
np.mat(ncenters)
ncenters = ncenters.T    #FCM模型中数据的输入需要转置一下
dim2 = ncenters.shape
print(dim2)
print('load data done')


#%%
u, u0, d, jm, p, fpc = cmeans_predict(testDspectraSWT, ncenters, m_t, error=0.005, maxiter=2000, seed = 0)
cluster_membership = np.argmax(u, axis=0)
results = {'fpc':fpc,'cluster_membership':cluster_membership,'u':u, 'd':d}
mat_path = analyDir + 'fcmresults5/FCM04cluster_{0}_preresults_in.mat'.format(m_t)
sio.savemat(mat_path,results)