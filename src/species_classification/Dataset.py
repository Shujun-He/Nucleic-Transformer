import pickle
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader



class Seq2SpeciesDataset(Dataset):
    def __init__(self,seqs,labels,transform=None):
        self.transform=transform
        self.seqs=seqs
        self.labels=labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        #sample = {'data': self.data[idx], 'labels': self.labels[idx]}
        sample = {'data': self.seqs[idx], 'labels': self.labels[idx]}
        if self.transform:
            sample=self.transform(sample)
        return sample
