import numpy as np
from attention_weights_visualizer import *
import pickle
import os
from tqdm import tqdm
import pandas as pd

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



os.system('mkdir aw_visualized')

count=0
sequences=[]
#for i in tqdm(range(len(prediction_dict['sequences']))):
for i in tqdm(range(558,559)):
    count+=1
    sequence=int2nucleotide(prediction_dict['sequences'][i])
    sequences.append(sequence)
    attention_weights=prediction_dict['attention_weights'][i]
    ground_truth=prediction_dict['ground_truths'][i]
    prediction=prediction_dict['predictions'][i]

    #nxn_attention_weights=np.asarray([k_mer_aggregated2nxn(attention_weights[i]) for i in range(len(attention_weights))])

    if ground_truth==1:
        state='positive'
    else:
        state='negative'

    if ground_truth==prediction:
        eval='correct'
    else:
        eval='wrong'

    for j,weights in enumerate(attention_weights):
        file="{}/{}_layer{}_{}_{}.png".format('aw_visualized',i+1,j,state,eval)
        plot_attention_weights(weights**4,sequence,sequence,file)


# df.sequence=np.asarray(sequences)
# df.index=np.arange(len(sequences))+1
# df.to_csv("aw_index.csv",index=False)
