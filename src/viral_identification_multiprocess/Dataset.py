import pickle
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader


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


class ViraminerDataset(Dataset):
    def __init__(self,df_path,transform=None):
        self.path=df_path
        self.transform=transform
        self._get_data_from_path()

    def _get_data_from_path(self):
        df=pd.read_csv(self.path)
        self.labels=df.iloc[:,2]
        self.data=[]
        for seq in df.iloc[:,1].to_list():
            self.data.append(nucleatide2int(seq))
        self.data=np.asarray(self.data,dtype='int64')
        self.labels=np.asarray(self.labels,dtype='int64')
        return self

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'labels': self.labels[idx]}
        if self.transform:
            sample=self.transform(sample)
        return sample
