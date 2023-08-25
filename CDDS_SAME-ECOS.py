#%% md

# Produce training data via SAME-ECOS simulation

#%%

import SAME_ECOS_functions_DWI as seD
import time
import datetime
import numpy as np
import scipy.io as sio
import math
import itertools

#%%

### load the decay library for 32-echo GRASE sequence
#    DWI_decay_lib = seD.load_decay_lib('DWIdecay_library_21bv3.mat')

#%%

### the decay library should have dimensions with 1st dim: D times (1-2000ms), 2nd dim: refocus flip angles (1-180 degrees), 3rd dim: echoes
DWI_decay_lib.shape

#%%
decay_library = sio.loadmat('DWIdecay_library_21bv3.mat')  #v3是倒数第二个版本，0.03,5 SNR40-300， d_basis50个
#decay_library = sio.loadmat('DWIdecay_library_21b-15v5.mat')
Dlist = decay_library['Dlist'].T
Dlist.shape
#%%

### use 'se.mp_yield_training_data' to accerelate the training data production by multiprocessing
### use 'se.produce_training_data' if encounter problems with the multiprocessing module
DWIdata = seD.mp_yield_training_data(seD.produce_training_data,
                                 DWI_decay_lib,
                                 Dlist,
                                 realizations=1000000,
                                 ncores=10, # number of cpu cores to use
                                 SNR_boundary_low=25,
                                 SNR_boundary_high=400,
                                 b_3=20/1000, # 3rd echo time is used to estimate the least plausible signal allowed in the simulation
                                 b_last=5000/1000,
                                 b_train_num=22,
                                 num_d_basis=40, # number of t2 basis to depict the simulated D spectrum
                                 peak_width=1, # the peak width of the simulated gaussian shaped D peaks
                                 D_min_universal=0.01,
                                 D_max_universal=3, # a default of 2000ms will be assigned if None is given
                                 exclude_M_max=True # exclude the maximum number of D components calculated from resolution limit
                                 )

#%%

DWIdata.keys()
#produced training data contains
# D_location: D peak locations of each realization
# D_locationID: id of the D_location
# D_amplitude: D peak amplitudes of each realization
# decay_curve: the pure signal at each echo of each realization
# decay_curve_with_noise: the signal at each echo of each realization with noise added
# train_label: the D spectrum (each D peak is depicted by two nearest basis t2s)
# train_label_gaussian: the D spectrum (each D peak is depicted by gaussian shaped basis t2s)
# num_D_SNR: a collection array of number of D peaks and selected SNR of each realization
#%%
import sklearn
import tensorflow as tf
import keras
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
#%%
x = DWIdata['decay_curve_with_noise']
### normalization the input data to its first echo (虽然DWI的曲线可能本身就是normalization过的
x = x/x[:,0].reshape(x.shape[0],1)
x.shape

#%%
y = DWIdata['train_label_gaussian']
y.shape
#%%   利用MATLAB检测过后的金标准进行训练，会不会提升准确率
import h5py
#%%
DWIdatafile = h5py.File('DWIdecay5_curve30upSNR300largerD_re2.mat','r')
DWIdata = {}
D_location_all = DWIdatafile['D_location_all'][:]
DWIdata['D_location'] = D_location_all.T
D_amplitude_all = DWIdatafile['D_amplitude_all'][:]
DWIdata['D_amplitude'] = D_amplitude_all.T
decay_curve_all = DWIdatafile['decay_curve_all'][:]
DWIdata['decay_curve'] = decay_curve_all.T
decay_curve_with_noise_all = DWIdatafile['decay_curve_with_noise_all'][:]
DWIdata['decay_curve_with_noise'] = decay_curve_with_noise_all.T

train_label_all = DWIdatafile['train_label_all'][:]
DWIdata['train_label'] = train_label_all.T
num_D_SNR_all = DWIdatafile['num_D_SNR_all'][:]
DWIdata['num_D_SNR'] = num_D_SNR_all.T

x = DWIdata['decay_curve_with_noise']
x = x/x[:,0].reshape(x.shape[0],1)
x.shape
y = DWIdata['train_label']
y.shape

#%% 新补充一些同样的峰值位置但是又100个不同噪声的曲线作为补充test
DWIdatafile = h5py.File('DWIdecay5_curve30upSNR300test2.mat','r')
DWIdata = {}

D_location_all_test = DWIdatafile['D_location_all_test2'][:]
DWIdata['D_location_test'] = D_location_all_test.T
D_amplitude_all_test = DWIdatafile['D_amplitude_all_test2'][:]
DWIdata['D_amplitude_test'] = D_amplitude_all_test.T
decay_curve_all_test = DWIdatafile['decay_curve_all_test2'][:]
DWIdata['decay_curve_test'] = decay_curve_all_test.T
decay_curve_with_noise_all_test = DWIdatafile['decay_curve_with_noise_all_test2'][:]
DWIdata['decay_curve_with_noise_test'] = decay_curve_with_noise_all_test.T
train_label_all_test = DWIdatafile['train_label_all_test2'][:]
DWIdata['train_label_test'] = train_label_all_test.T
num_D_SNR_all_test = DWIdatafile['num_D_SNR_all_test2'][:]
DWIdata['num_D_SNR_test'] = num_D_SNR_all_test.T
#%%
DWIdatafile = h5py.File('DWIdecay5_curve50upSNR300test2.mat','r')
D_location_all = DWIdatafile['D_location_all'][:]
DWIdata['D_location'] = D_location_all.T
DWIdata['D_location'].shape
#%% split the dataset into training, validation, and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=10)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=10)
train_ID, testID = train_test_split(np.arange(0, 5000000), test_size=0.1, random_state=10)

print('training x: ' + str(x_train.shape))
print('training y: ' + str(y_train.shape))
print('validation x: ' + str(x_val.shape))
print('validation y: ' + str(y_val.shape))
print('testing x: ' + str(x_test.shape))
print('testing y: ' + str(y_test.shape))

#%%
### Setting up neural network hyperparameters
hidden_layers = [100,200,500,1000,1000,500,200]
output_layer_nodes = y.shape[1]
Batch_norm = 'yes'
acti = 'selu'
initial = 'lecun_normal'
selected_metrics = [tf.keras.metrics.CosineSimilarity(axis=-1)]
selected_optimizer = keras.optimizers.Adamax(lr=0.002)
# loss_function = 'cosine_similarity'
loss_function = 'categorical_crossentropy' # other loss functions can be used as well.
# loss_function = 'mean_squared_error'
l1_strength = 0.001 # L1 regularization parameter
l2_strength = 0.001 # L2 regularization parameter
num_epoch = 40

#%%
### Train and test the neural network model, and store training and testing history in 'NN_training.txt'
f = open("NN_trainingDWI14_3.txt", "a")
now = datetime.datetime.now()
f.write("Date and Time: {} \n".format(now.strftime("%Y-%m-%d %H:%M:%S")))
f.write('selected hidden layer structure = {} \n'.format(hidden_layers))
f.write('activation = {}, initialization = {} \n'.format(acti, initial))
f.write('Batch_norm = {}, loss_function = {}, regularization_strength = {}, \nselected_optimizer = {} \n \n \n'.format(Batch_norm, loss_function, l2_strength, selected_optimizer))

keras.backend.clear_session()
model = keras.Sequential()
if Batch_norm == 'yes':
    model.add(keras.layers.BatchNormalization())
for nodes in hidden_layers:
    model.add(keras.layers.Dense(nodes, kernel_initializer=initial, activation=acti))
    if Batch_norm == 'yes':
        model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(output_layer_nodes, kernel_initializer=initial, activation='softmax', activity_regularizer=regularizers.l1_l2(l1=l1_strength, l2=l2_strength)))
model.compile(optimizer=selected_optimizer, loss = loss_function, metrics = selected_metrics)

callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_cosine_similarity', patience=20), #原代码是patience=15
                  keras.callbacks.ModelCheckpoint(filepath='NN_model_example_DWI.h5', monitor='val_cosine_similarity', save_best_only=True),
                  keras.callbacks.ReduceLROnPlateau(monitor='cosine_similarity', factor=0.5, patience=3)]

start_time = time.time()
history = model.fit(x_train, y_train, epochs=40, batch_size = 1280, callbacks = callbacks_list, validation_data=(x_val, y_val))
# 原代码 batch_size = 256
print("--- %.2f seconds ---" %(time.time() - start_time))
f.write("--- Training time %.2f seconds --- \n" %(time.time()-start_time))
print('training stopped at epoch = {}'.format(len(history.history['lr'])))
f.write('training stopped at epoch = {} \n \n'.format(len(history.history['lr'])))
f.write('validation similarity = {} \n \n'.format(list(round(i, 4) for i in history.history['val_cosine_similarity'])))
f.write('learning rate = {} \n \n'.format(list(round(i,6) for i in history.history['lr'])))

### apply trained model to the test set
NN_predict = model.predict(x_test)

### evaluate the model performance using cosine similarity scores
similarity_score = np.zeros((y_test.shape[0],1))
for item in range(y_test.shape[0]):
    similarity_score[item,:] = cosine_similarity(NN_predict[item,:].reshape(1,-1), y_test[item,:].reshape(1,-1))
print('Average similarity for each spectrum in the test set = {} +/- {} \n'.format(similarity_score.mean(), similarity_score.std()))
f.write('Average similarity for each spectrum in the test set = {} +/- {} \n'.format(similarity_score.mean(), similarity_score.std()))
f.write('#######################################################\n')
f.write('\n \n')
f.close()


#%% md
# Evaluate the trained model performance
import matplotlib.pyplot as plt
#%%
d_basis_o = DWIdatafile['d_basis'][:]
d_basis = d_basis_o.T
d_basis = np.array(d_basis)
d_basis = d_basis.reshape(50,1)
DWIdatafile.close()
#%%
### randomly pick a few test examples and plot them
import joblib
import os
from keras.models import load_model
model = load_model('NN_model_DWI14_v4.h5')
x_test2 = DWIdata['decay_curve_with_noise_test']
x_test2 = x_test2/x_test2[:,0].reshape(x_test2.shape[0],1)
x_test2.shape
y_test2 = DWIdata['train_label_test']
y_test2.shape
NN_predict = model.predict(x_test2)

#%%
similarity_score_test = np.zeros((y_test2.shape[0],1))
for item in range(y_test2.shape[0]):
    similarity_score_test[item,:] = cosine_similarity(NN_predict[item,:].reshape(1,-1), y_test2[item,:].reshape(1,-1))

#%%
nrow = 3
ncol = 3
plt.figure(figsize=(25,10))
plt.style.use('ggplot')
num_D_SNR_copy = DWIdata['num_D_SNR']
for i in range(nrow*ncol):
    plt.subplot(nrow,ncol, i+1)
    #random_pick = np.random.randint(0,y_test.shape[0])
    #testID_pick = testID[random_pick]
    #plt.plot(d_basis, NN_predict[random_pick,:], 'r', label='prediction')
    plt.plot(d_basis, NN_validation[i, :], 'r', label='validation')
    #plt.plot(d_basis, y_test[random_pick,:], 'b', label='ground truth')
    plt.plot(d_basis, y_validation[i,:],'b', label='ground truth')
    plt.legend(fontsize=12)
    plt.xscale('log')
    #plt.title(['Similarity = {}'.format(similarity_score[random_pick])+'SNR={}'.format(num_D_SNR_copy[testID_pick,1])], fontsize=15)
    plt.title(['Similarity = {}'.format(similarity_score_validation[i]) + 'SNR={}'.format(num_D_SNR_copy[i, 1])],fontsize=15)
plt.show()
#%%
nrow = 1
ncol = 2
plt.figure(figsize=(15,10))
plt.style.use('ggplot')
num_D_SNR_copy = DWIdata['num_D_SNR']
for i in range(nrow*ncol):
    plt.subplot(nrow,ncol, i+1)
    plt.plot(d_basis, NN_validation[i,:], 'r', label='validation')
    plt.plot(d_basis, y_validation[i,:], 'b', label='ground truth')
    plt.legend(fontsize=12)
    plt.xscale('log')
    plt.title(['Similarity = {}'.format(similarity_score_test[i])+'SNR={}'.format(num_D_SNR_copy[i,1])], fontsize=15)
plt.show()
#%%   paper中作图——新test峰值个数分别为1,2,3不同噪声下的相似度展示
from matplotlib import font_manager
nrow = 1
ncol = 4
plt.figure(figsize=(35,6))
plt.style.use('ggplot')
num_D_SNR_copy_test = DWIdata['num_D_SNR_test']
plt.subplot(nrow,ncol,1)  #one peak
for i in range(99):
    plt.plot(d_basis,NN_predict[i,:], color='gray',linewidth=1)
plt.plot(d_basis,NN_predict[99,:], color='gray',linewidth =1,label='SAME-ECOS')
plt.plot(d_basis,np.mean(NN_predict[0:100,:],axis=0),color = 'r',linewidth=3, label='SAME-ECOS (mean)')
plt.plot(d_basis,np.mean(y_test2[0:100,:],axis=0),color = 'g', linewidth=2, label='ground truth')
font2 = {'family':'Times New Roman','weight':'normal','size':12}
plt.legend(loc='upper right',ncol=1,prop=font2)
plt.xscale('log')
font1 = {'family':'Times New Roman','weight':'normal','size':15}
plt.title(['Similarity = {}'.format(round(np.mean(similarity_score_test[0:100,:]),4))], x = 0.2, y=0.9,fontdict={'family':'Times New Roman','size':15})

plt.subplot(nrow,ncol,2)
for i in range(100,199):
    plt.plot(d_basis,NN_predict[i,:], color='gray',linewidth=1)
plt.plot(d_basis,NN_predict[199,:], color='gray',linewidth=1,label='SAME-ECOS')
plt.plot(d_basis,np.mean(NN_predict[100:200,:],axis=0),color = 'r',linewidth=3, label='SAME-ECOS (mean)')
plt.plot(d_basis,np.mean(y_test2[100:200,:],axis=0),color = 'g', linewidth=2, label='ground truth')
font2 = {'family':'Times New Roman','weight':'normal','size':12}
plt.legend(loc='upper right',ncol=1,prop=font2)
plt.xscale('log')
font1 = {'family':'Times New Roman','weight':'normal','size':15}
plt.title(['Similarity = {}'.format(round(np.mean(similarity_score_test[100:200,:]),4))], x = 0.2, y=0.9,fontdict={'family':'Times New Roman','size':15})

plt.subplot(nrow,ncol,3)
for i in range(400,499):
    plt.plot(d_basis,NN_predict[i,:], color='gray',linewidth=1)
plt.plot(d_basis,NN_predict[499,:], color='gray',linewidth=1, label = 'SAME-ECOS')
plt.plot(d_basis,np.mean(NN_predict[400:500,:],axis=0),color = 'r',linewidth=3, label='SAME-ECOS (mean)')
plt.plot(d_basis,np.mean(y_test2[400:500,:],axis=0),color = 'g', linewidth=2, label='ground truth')
font2 = {'family':'Times New Roman','weight':'normal','size':12}
plt.legend(loc='upper right',ncol=1,prop=font2)
plt.xscale('log')
font1 = {'family':'Times New Roman','weight':'normal','size':15}
plt.title(['Similarity = {}'.format(round(np.mean(similarity_score_test[400:500,:]),4))], x = 0.2, y=0.9,fontdict={'family':'Times New Roman','size':15})

plt.subplot(nrow,ncol,4)
for i in range(300,399):
    plt.plot(d_basis,NN_predict[i,:], color='gray',linewidth=1)
plt.plot(d_basis,NN_predict[399,:], color='gray',linewidth=1, label = 'SAME-ECOS')
plt.plot(d_basis,np.mean(NN_predict[300:400,:],axis=0),color = 'r',linewidth=3, label='SAME-ECOS (mean)')
plt.plot(d_basis,np.mean(y_test2[300:400,:],axis=0),color = 'g', linewidth=2, label='ground truth')
font2 = {'family':'Times New Roman','weight':'normal','size':12}
plt.legend(loc='upper right',ncol=1,prop=font2)
plt.xscale('log')
font1 = {'family':'Times New Roman','weight':'normal','size':15}
plt.title(['Similarity = {}'.format(round(np.mean(similarity_score_test[300:400,:]),4))], x = 0.2, y=0.9,fontdict={'family':'Times New Roman','size':15})
plt.savefig("D:/PERSONAL/goonphd/DWImodel193rd/brain_data/Dspectrum/results_pictures/SAME-ECOStestsimilarity.pdf",bbox_inches = 'tight',dpi=600)

plt.show()
#%%
model.save('NN_model_DWI14_2_v4.h5')   #10_6  0.87579但是好像没保存好，  就用10_9作为模型吧，不换了！！！！！！
#倒数第二版的模型用的是14， d_basis用的50个
#v5代表d_basis0.06-8,80个，其余与14_v4一致
#v6代表d_basis0.06-15,80个，其余与15_v5一致
# 16_1代表用了100个d_basis，其余同上
# 15_1代表d_basis0.03-8,60个，其余与15_v5一致
# 17_v7代表d_basis0.03-10,50个，其余与15_1一致
# 17_1_v7代表d_basis0.03-15,60个，包含了14的d_basis,其他与15一致
# 17_2_v7代表d_basis0.03-15,60个，但decay_curve删掉了第一个b=0的数据，其他同17_1
#%%
# Apply trained model to experimental data
analyDir = "D:/PERSONAL/goonphd/DWImodel193rd/brain_data/Dspectrum/Huashan/"
patientname = "patientname"
Ddecay_lib = sio.loadmat(analyDir+patientname+"/Ddecaycurve.mat")
Ddecay_curve = Ddecay_lib['normdecay']
# 尝试删掉b=0的值，看一下脑脊液那块会不会还会有边缘很大的凸起
Ddecay_curve = np.delete(Ddecay_curve.copy(),0,1)
Ddecay_curve.shape
#%%
NN_predict_vivo = model.predict(Ddecay_curve)
nrow = 3
ncol = 4
plt.figure(figsize=(25,10))
plt.style.use('ggplot')
for i in range(nrow*ncol):
    plt.subplot(nrow,ncol, i+1)
    random_pick = np.random.randint(0,Ddecay_curve.shape[0])
    #testID_pick = testID[random_pick]
    plt.plot(d_basis, NN_predict_vivo[random_pick,:], 'b', label='vivo prediction')
    #plt.plot(d_basis, y_test[random_pick,:], 'b', label='ground truth')
    plt.legend(fontsize=12)
    plt.xscale('log')
    plt.title([random_pick], fontsize=15)
plt.show()
#%%----------找几个图和15-v5对比一下
picknum = [41341,415998,127251,121933,38497,33930,405124,383404,72357,365076,144621,28707]
nrow = 3
ncol = 4
plt.figure(figsize=(25,10))
plt.style.use('ggplot')
for i in range(nrow*ncol):
    plt.subplot(nrow,ncol, i+1)
    random_pick = picknum[i]
    #testID_pick = testID[random_pick]
    plt.plot(d_basis, NN_predict_vivo[random_pick,:], 'b', label='vivo prediction')
    #plt.plot(d_basis, y_test[random_pick,:], 'b', label='ground truth')
    plt.legend(fontsize=12)
    plt.xscale('log')
    plt.title([random_pick], fontsize=15)
plt.show()
#%%------------再看一些其他的体素比较一下
picknum = [70232,107432,115148,5805,338206,360976,217869,84165,48568,359554,71072,173818]
nrow = 3
ncol = 4
plt.figure(figsize=(25,10))
plt.style.use('ggplot')
for i in range(nrow*ncol):
    plt.subplot(nrow,ncol, i+1)
    random_pick = picknum[i]
    #testID_pick = testID[random_pick]
    plt.plot(d_basis, NN_predict_vivo[random_pick,:], 'b', label='vivo prediction')
    #plt.plot(d_basis, y_test[random_pick,:], 'b', label='ground truth')
    plt.legend(fontsize=12)
    plt.xscale('log')
    plt.title([random_pick], fontsize=15)
plt.show()
#%%---------遍历整个文件夹68个病人，计算出他们体素的D谱并存储/----------------------------------------
# 还有遍历6个健康对照组的voxellevel的谱线
# 还有基于ROI averaged Ddecay的谱线
import os
import joblib
import joblib
import os
from keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
model = load_model('NN_model_DWI14_v4.h5')
#analyDir = "D:/PERSONAL/goonphd/DWImodel193rd/brain_data/Dspectrum/Huashan/"
#analyDir = "D:/PERSONAL/goonphd/DWImodel193rd/brain_data/Dspectrum/Huashan_ctrl/"

analyDir = "D:/PERSONAL/goonphd/DWImodel193rd/brain_data/Dspectrum/Huashan_add/"
for patient in os.listdir(analyDir):
    patientDir = os.path.join(analyDir,patient)
    if os.path.isfile(patientDir+"/Wholebrain_Dspectrum_v4.pkl"):
        continue
    Ddecay_lib = sio.loadmat(patientDir + "/Ddecaycurve.mat")
    Ddecay_curve = Ddecay_lib['normdecay']
    #Ddecay_lib = sio.loadmat(patientDir + "/ROIDdecay.mat")
    #Ddecay_curve = Ddecay_lib['ROIDdecay']
    #都删掉b=0的那一项的值看看
    #Ddecay_curve = np.delete(Ddecay_curve.copy(),0,1)


    NN_predict_vivo = model.predict(Ddecay_curve)
    #joblib.dump(NN_predict_vivo,patientDir + "/ROIaver_Dspectra_v4.pkl")  #最终没有计算v4_2
    joblib.dump(NN_predict_vivo,patientDir + "/Wholebrain_Dspectrum_v4.pkl")
    print(patient)


#%% --------------------------MATLAB计算矩阵会更快，把pkl转换成mat文件-------------------------
import scipy.io as scio
import os
#analyDir = 'D:/PERSONAL/goonphd/DWImodel193rd/brain_data/Dspectrum/Huashan/'
#analyDir = 'D:/PERSONAL/goonphd/DWImodel193rd/brain_data/Dspectrum/Huashan_ctrl/'
analyDir = 'D:/PERSONAL/goonphd/DWImodel193rd/brain_data/Dspectrum/Huashan_add/'
for patient in os.listdir(analyDir):
    patientDir = os.path.join(analyDir,patient)
    NN_predict_vivo = joblib.load(patientDir + "/Wholebrain_Dspectrum_v4.pkl")
    #NN_predict_vivo = joblib.load(patientDir + "/ROIaver_Dspectra_v4.pkl")
    NN_predict_vivo = np.mat(NN_predict_vivo)
    scio.savemat(patientDir + '/Wholebrain_Dspectrum_v4.mat',{'NN_predict_vivo':NN_predict_vivo})
    #scio.savemat(patientDir + '/ROIaver_Dspectra_v4.mat', {'NN_predict_vivo': NN_predict_vivo})
#%%
#Ddecay_lib = sio.loadmat("D:/PERSONAL/goonphd/DWImodel193rd/brain_data/Dspectrum/ROIDdecayall.mat")
Ddecay_lib = sio.loadmat("D:/PERSONAL/goonphd/DWImodel193rd/brain_data/Dspectrum/ROIDdecayallcontrols.mat")
Ddecay_curve = Ddecay_lib['ROIDdecayall']
NN_predict_vivo = model.predict(Ddecay_curve)
NN_predict_vivo = np.mat(NN_predict_vivo)
scio.savemat('D:/PERSONAL/goonphd/DWImodel193rd/brain_data/Dspectrum/ROIaver_Dspectraallcontrols_v4.mat', {'NN_predict_vivo': NN_predict_vivo})
#%%
import scipy.io as scio
import os
scio.savemat('D:/PERSONAL/goonphd/myPython3/SAME-ECOS-master/testpre_v5.mat',{'NN_predict_test':NN_predict,'similarity_score_test':similarity_score_test})
#%%  查看vivo 数据的计算结果
import os
import joblib
analyDir = "D:/PERSONAL/goonphd/DWImodel193rd/brain_data/Dspectrum/Huashan/"
patientname = "CAI_ZHONG_GUO"
NN_predict_vivo  = joblib.load(analyDir + patientname + "/Wholebrain_Dspectrum_v4.pkl")
DWIdatafile = h5py.File('DWIdecay5_curve50upSNR300largerD.mat')
d_basis_o = DWIdatafile['d_basis'][:]
d_basis = d_basis_o.T
d_basis = np.array(d_basis)
d_basis = d_basis.reshape(50,1)
Ddecay_lib = sio.loadmat(analyDir +patientname + "/Ddecaycurve.mat")
Ddecay_curve = Ddecay_lib['normdecay']
nrow = 3
ncol = 4
plt.figure(figsize=(25,10))
plt.style.use('ggplot')
for i in range(nrow*ncol):
    plt.subplot(nrow,ncol, i+1)
    random_pick = np.random.randint(0,Ddecay_curve.shape[0])
    #testID_pick = testID[random_pick]
    plt.plot(d_basis, NN_predict_vivo[random_pick,:], 'b', label='vivo prediction')
    #plt.plot(d_basis, y_test[random_pick,:], 'b', label='ground truth')
    plt.legend(fontsize=12)
    plt.xscale('log')
    plt.title([random_pick], fontsize=15)
plt.show()
