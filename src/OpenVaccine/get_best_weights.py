import numpy as np
import os
import pandas as pd



def get_best_weights_from_fold(fold,top=5):
    csv_file='log_fold{}.csv'.format(fold)

    history=pd.read_csv(csv_file)
    scores=np.asarray(-history.val_loss)
    top_epochs=scores.argsort()[-top:][::-1]
    print(scores[top_epochs])
    os.system('mkdir best_weights')

    for i in range(top):
        weights_path='checkpoints_fold{}/epoch{}.ckpt'.format(fold,history.epoch[top_epochs[i]])
        print(weights_path)
        os.system('cp {} best_weights/fold{}top{}.ckpt'.format(weights_path,fold,i+1))

    return scores[top_epochs[0]]

scores=[]
for i in range(10):
    scores.append(get_best_weights_from_fold(i))

with open('cv.txt','w+') as f:
    f.write(str(-np.mean(scores)))
