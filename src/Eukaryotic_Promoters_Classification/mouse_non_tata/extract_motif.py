import numpy as np
import pickle
import os
from tqdm import tqdm
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 10}

matplotlib.rc('font', **font)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kmers', type=int, default='7',  help='kmer')
    opts = parser.parse_args()
    return opts

opts=get_args()

nts=[
"A",
"T",
"G",
"C"]

def int2nucleotide(nt_sequence,target_length=None):
    seq=''
    for nt in nt_sequence:
        seq+=nts[nt]
    return seq

with open("prediction_dict.p","rb") as f:
    prediction_dict=pickle.load(f)


df=pd.DataFrame(columns=['index','sequence'])

def get_kmers(sequence,k):
    kmers=[]
    for i in range(len(sequence)-k+1):
        kmers.append(sequence[i:i+k])
    return kmers

os.system('mkdir aw_visualized')

top=10
count=0
sequences=[]
top_kmers=[]
top_k_count=[]
for i in tqdm(range(len(prediction_dict['sequences']))):

    count+=1
    sequence=int2nucleotide(prediction_dict['sequences'][i])
    sequences.append(sequence)
    attention_weights=prediction_dict['attention_weights'][i]
    ground_truth=prediction_dict['ground_truths'][i]
    prediction=prediction_dict['predictions'][i]

    kmers=np.asarray(get_kmers(sequence,opts.kmers))

    attention_weights=attention_weights[-1].sum(0)
    #attention_weights=attention_weights/attention_weights.sum()
    # plt.imshow(attention_weights.reshape(1,-1).astype('float32'))
    # plt.show()
    #exit()
    if ground_truth==1:
        state='positive'
    else:
        state='negative'

    if ground_truth==prediction:
        eval='correct'
    else:
        eval='wrong'
    if state=='positive' and eval=='correct':
        sorted_indices=np.argsort(attention_weights)
        #print(attention_weights[sorted_indices][-3:])
        top_k=kmers[sorted_indices][-3:]
        for kmer in top_k:
            if kmer not in top_kmers:
                top_kmers.append(kmer)
                top_k_count.append(1)
            else:
                top_k_count[top_kmers.index(kmer)]=top_k_count[top_kmers.index(kmer)]+1
    #exit()

top_kmers=np.asarray(top_kmers)
top_k_count=np.asarray(top_k_count)

#exit()

top_indices=np.flip(np.argsort(top_k_count))

fig, ax = plt.subplots()
x=np.arange(top)
width=0.4
bar=ax.bar(x,top_k_count[top_indices[:top]],edgecolor='k',linewidth=2)
ax.set_ylabel('Num of appearancesin top 3',fontsize=10)
#ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(top_kmers[top_indices[:top]])
plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
         rotation_mode="anchor")
ax.legend()
plt.savefig('promoter_motifs.eps')
#plt.show()
