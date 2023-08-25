#%%
import os
import sys
sys.path.append("/home3/")
import numpy as np
import matplotlib.pyplot as plt
from skfuzzy.cluster import cmeans
import scipy.io as sio
from multiprocessing import Pool as ThreadPool
import joblib
import h5py
import math
print('Import lib done')

#%%
# prepare data

analyDir = '/home3/Dspectraanaly/'


matfn = analyDir + 'trainDspectraWT_all.mat'
data = h5py.File(matfn)
type(data)
g4DspectraNET = data['trainT2Dsp'][:]
#np.array(trainDspectraWT, dtype = 'float32')
np.mat(g4DspectraNET)
#trainDspectraWT = trainDspectraWT.T    #FCM模型中数据的输入需要转置一下
dim = g4DspectraNET.shape

m_t = 1 + (1418/dim[1]+22.05)*(dim[0]**(-2))+(12.33/dim[1]+0.243)*(dim[0]**(-0.0406*math.log(dim[1])-0.1134))
print('load data done')
print(dim)


#%% Now we can use FCM to cluster the curves
# 1、data就是训练的数据。这里需要注意data的数据格式，shape是类似（特征数目，数据个数），与很多训练数据的shape正好是相反的。
#2、c是需要指定的聚类个数。
#3、m也就是上面提到的隶属度的指数，是一个加权指数。
#4、error就是当隶属度的变化小于此的时候提前结束迭代。
#5、maxiter最大迭代次数。

#1、cntr聚类的中心。
#2、u是最后的的隶属度矩阵。
#3、u0是初始化的隶属度矩阵。
#4、d是最终的每个数据点到各个中心的欧式距离矩阵。
#5、jm是目标函数优化的历史。
#6、p是迭代的次数。
#7、fpc全称是fuzzy partition coefficient，是一个评价分类好坏的指标。它的范围是0到1，1是效果最好。后面可以通过它来选择聚类的个数。
#%%
nclusters = [2,3,4,5,6]
ms = [m_t,1.2,2]
n = 0
combs = {}
for i in range(len(nclusters)):
    for j in range(len(ms)):
        combs[n] = (nclusters[i],ms[j])
        n = n + 1 
items = [i for i in range(len(nclusters)*len(ms))]
#%%
#combs = {'1':(2,1.0739), '2':(5,1.0739), '3':(3,1.0739), '4':(4,1.0739), '5':(6,1.0739), '6':(7,1.0739), '7':(8,1.0739), '8':(9,1.0739), '9':(10,1.0739), '10':(15,1.0739), '11':(20,1.0739), '12':(25,1.0739),'13':(26,1.0739),'14':(27,1.0739),'15':(28,1.0739),'16':(29,1.0739),'17':(30,1.0739)}
#for ax, ncluster in enumerate(nclusters):
#items = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17']
def processing(item):
    comb = combs[item]
    n_cluster = comb[0]
    m = comb[1]
    fpcs = {}
    cluster_memberships = {}
    centers = {}
    ps = {}
    jms = {}
    us = {}
    ds = {}
    center, u, u0, d, jm, p, fpc = cmeans(g4DspectraNET, n_cluster, m, error=0.005, maxiter=2000, seed = 0)
    fpcs[(n_cluster, m)] = fpc
    us[(n_cluster, m)] = u
    cluster_membership = np.argmax(u, axis=0)
    cluster_memberships[(n_cluster, m)] = cluster_membership
    centers[(n_cluster, m)] = center
    ps[(n_cluster, m)] = p
    jms[(n_cluster, m)] = jm
    ds[(n_cluster, m)] = d
    results = {'fpcs':fpcs, 'us':us, 'cluster_memberships':cluster_memberships, 'centers':centers,'jms':jms,'d':ds}
    print("%s has been done" % ((n_cluster, m),))
    return (results)

#%%
pool = ThreadPool()
multiresults = pool.map(processing, items)
pool.close()
pool.join()
# %%

#----------------------------------------------save all trained vatiables-----------------------------------------------



# Each variable in 'FCM_gliom_init_varbs' is a dict contains key(ncluster,m) with its compareble values
joblib.dump(multiresults,analyDir + 'fcmresults5/FCM_trainDspectraNET_all.pkl')

