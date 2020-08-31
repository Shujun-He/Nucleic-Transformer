import pandas as pd
import numpy as np

nfolds=5
gap=25
scores=[]
for fold in range(nfolds):
    df=pd.read_csv("log_fold{}.csv".format(fold))
    scores.append(df.val_acc.to_list())
    
scores=np.asarray(scores)
scores=np.mean(scores.reshape(nfolds,-1,gap),axis=-1)
print(np.mean(scores,axis=0))  
print(max(np.mean(scores,axis=0)))
  
# with open("cv_avg.txt",'w+') as f:
    # f.write("{}-fold score: {}".format(nfolds,np.mean(scores)))
    