import pickle
from sklearn import metrics
from tqdm import tqdm
import numpy as np


with open('test_results.p','rb') as f:
    outputs,ground_truths=pickle.load(f)

aucs=[]
for i in tqdm(range(598)):
    auc=metrics.roc_auc_score(ground_truths[:,i],outputs[:,i])
    aucs.append(auc)

aucs.append(1)
for i in tqdm(range(599,919)):
    auc=metrics.roc_auc_score(ground_truths[:,i],outputs[:,i])
    aucs.append(auc)

import pandas as pd
df=pd.DataFrame(columns=['AUC'])
df['AUC']=aucs

df.to_csv('test_aucs.csv')

# #exit()
# aucs=[]
# for i in tqdm(range(125)):
#     auc=metrics.roc_auc_score(ground_truths[:,i],outputs[:,i])
#     aucs.append(auc)
#     all_aucs.append(auc)
#
# DNase_median_acu=np.median(aucs)
#
#
# aucs=[]
# for i in tqdm(range(125,598)):
#     auc=metrics.roc_auc_score(ground_truths[:,i],outputs[:,i])
#     aucs.append(auc)
#
#
# for i in tqdm(range(599,815)):
#     auc=metrics.roc_auc_score(ground_truths[:,i],outputs[:,i])
#     aucs.append(auc)
#
# TF_median_acu=np.median(aucs)
#
# aucs=[]
# for i in tqdm(range(815,919)):
#     auc=metrics.roc_auc_score(ground_truths[:,i],outputs[:,i])
#     aucs.append(auc)
#
#
# Histone_median_acu=np.median(aucs)
#
# with open("test_results.txt",'w+') as f:
#     f.write(f"DNase_median_acu: {DNase_median_acu}\n")
#     f.write(f"TF_median_acu: {TF_median_acu}\n")
#     f.write(f"Histone_median_acu: {Histone_median_acu}\n")
