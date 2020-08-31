import pickle
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.model_selection import StratifiedKFold


nt_int={
"A": 0,
"T": 1,
"G": 2,
"C": 3,}

def nucleatide2int(nt_sequence,target_length=None):
    int_sequence=[]
    for nt in nt_sequence:
        nt=nt.upper()
        if nt in nt_int:
            int_sequence.append(nt_int[nt])
    int_sequence=np.asarray(int_sequence,dtype='int32')
    if target_length:
        int_sequence=np.pad(int_sequence,(0,target_length-len(int_sequence)),constant_values=-1)
    return int_sequence


class ViraMiner_Dataset(torch.utils.data.Dataset):
    def __init__(self,path,fold,nfolds=5,training=True,shuffle=True,batch_size=32,seed=7,factor=5):
        self.path=path
        self.fold=fold
        self.nfolds=nfolds
        self.training=training
        self.shuffle=shuffle
        self.batch_size=batch_size
        self.seed=seed
        self.factor=factor
        self._get_data_from_path()
        self._update_len()

    def switch_mode(self,training):
        if training:
            self.training=True
        else:
            self.training=False
        return self

    def update_batchsize(self,batch_size):
        self.batch_size=batch_size
        self._update_len()
        return self

    def _update_len(self):
        if len(self.train_indices)%self.batch_size==0:
            self.train_batches=int(len(self.train_indices)/self.batch_size)
        else:
            self.train_batches=int(len(self.train_indices)/self.batch_size)+1
        if len(self.val_indices)%self.batch_size==0:
            self.val_batches=int(len(self.val_indices)/self.batch_size)
        else:
            self.val_batches=int(len(self.val_indices)/self.batch_size)+1
        return self

    def __len__(self):
        if self.training:
            return self.train_batches
        else:
            return self.val_batches

    def _get_train_indices(self,):
        # negative_samples=np.random.choice(len(self.negative_train_indices),self.factor*len(self.positive_train_indices),replace=False)
        # self.train_indices=[self.negative_train_indices[negative_samples],self.positive_train_indices]
        # self.train_indices=np.concatenate(self.train_indices)
        self.train_indices=np.arange(len(self.train_df))
        np.random.shuffle(self.train_indices)
        self._update_len()
        print("###train indices shuffled###")

    def __getitem__(self,idx):
        if idx==0 and self.training and self.shuffle:
            self._get_train_indices()
        if self.training:
            indices=self.train_indices[idx*self.batch_size:(idx+1)*self.batch_size]
        else:
            indices=self.val_indices[idx*self.batch_size:(idx+1)*self.batch_size]

        data=self.data[indices]
        labels=self.labels[indices]
        directions=np.zeros(len(indices))
        #labels[labels==2]=1
        return {'data':data,'labels':labels,'directions':directions}

    def _get_data_from_path(self):
        '''
        nucleotide_types:
            padding=-1
            A=0
            T=1
            G=2
            C=3

        labels:
            non promoter=0
            promoter=1
        '''

        train=os.path.join(self.path,"fullset_train.csv")
        val=os.path.join(self.path,"fullset_validation.csv")
        test=os.path.join(self.path,"fullset_test.csv")
        train_df=pd.read_csv(train)
        self.train_df=train_df
        self.labels=train_df.iloc[:,2]
        self.data=[]
        for seq in train_df.iloc[:,1].to_list():
            self.data.append(nucleatide2int(seq))
        val_df=pd.read_csv(val)
        for seq in val_df.iloc[:,1].to_list():
            self.data.append(nucleatide2int(seq))

        #print(self.data)
        self.data=np.asarray(self.data)
        self.labels=np.asarray(self.labels,dtype='int64')

        self.train_indices=np.arange(len(train_df))
        self.positive_train_indices=self.train_indices[self.labels==1]
        self.negative_train_indices=self.train_indices[self.labels==0]
        self.labels=np.concatenate([self.labels,val_df.iloc[:,2]])
        self.val_indices=np.arange(len(train_df),len(train_df)+len(val_df))
        self._get_train_indices()
        return self

    def _get_longest_sequence_length(self,a,b,c):
        self.max_length=0
        with open(a,'r') as f:
            for line in f:
                if len(line)>self.max_length:
                    self.max_length=len(line)
        with open(b,'r') as f:
            for line in f:
                if len(line)>self.max_length:
                    self.max_length=len(line)
        with open(c,'r') as f:
            for line in f:
                if len(line)>self.max_length:
                    self.max_length=len(line)
