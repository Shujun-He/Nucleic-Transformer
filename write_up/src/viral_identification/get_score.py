import pandas as pd
import numpy as np

nfolds=5
scores=[]
for fold in range(nfolds):
    try:
        df=pd.read_csv("log_fold{}.csv".format(fold))
        #scores.append(max(df.iloc[:,2].to_list()))
        scores.append(max(df.val_acc.to_list()))
    except:
        pass

print(np.mean(scores))

with open("cv_avg.txt",'w+') as f:
    f.write("{}-fold score: {}".format(nfolds,np.mean(scores)))
