import sys
sys.path.append("/home3/FUNC")

import numpy as np
import joblib
import scipy.io as io
import os


analyDir = '/home3/'
multiresults = joblib.load(analyDir + 'fcmresults5/FCM_trainDspectraNET_all.pkl')
for i in range(len(multiresults)):
    data = multiresults[i]
    fpcs = data['fpcs']
    for k in fpcs.keys():
        k = k
    us = data['us']
    cluster_memberships = data['cluster_memberships']
    centers = data['centers']
    #ps = data['ps']
    jms = data['jms']
    ds = data['d']
    results = {'fpcs':fpcs[k], 'us':us[k], 'cluster_memberships':cluster_memberships[k], 'centers':centers[k], 'jms':jms[k]}
    if k[0]<10:
    	mat_path = analyDir +'fcmresults5/NET_all/FCM0{0}cluster_{1}_results.mat'.format(k[0],k[1])
    else:

    	mat_path = analyDir + 'fcmresults5/NET_all/FCM{0}cluster_{1}_results.mat'.format(k[0],k[1])
    io.savemat(mat_path,results)