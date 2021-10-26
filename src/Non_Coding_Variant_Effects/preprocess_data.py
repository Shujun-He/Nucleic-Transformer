import scipy.io
import h5py
import numpy as np

f=h5py.File('deepsea_train/train.mat', 'r')# as f:
train_seqs =np.array(f['trainxdata']).transpose(2,1,0).astype('uint8')
train_labels =np.array(f['traindata']).transpose(1,0).astype('uint8')
val_data = scipy.io.loadmat('deepsea_train/valid.mat')
val_seqs = np.array(val_data['validxdata']).transpose(2,1,0).astype('uint8')
val_labels = np.array(val_data['validdata']).transpose(1,0).astype('uint8')

import pickle
with open('DeepSea_TrainVal.p','wb+') as f:
    pickle.dump([train_seqs,train_labels,val_seqs,val_labels],f)
