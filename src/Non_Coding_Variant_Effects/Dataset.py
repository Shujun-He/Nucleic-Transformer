import pickle
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch


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


class DeepSeaDataset(torch.utils.data.Dataset):
    def __init__(self,sequences,labels):
        self.sequences=sequences
        self.labels=labels


    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        sequence=self.sequences[idx].argmax(0)
        return {'data':sequence, 'labels':self.labels[idx]}
