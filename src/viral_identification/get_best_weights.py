import numpy as np
import os
import pandas as pd



def get_best_weights_from_fold(fold,top=5):
    csv_file='log_fold{}.csv'.format(fold)

    history=pd.read_csv(csv_file)
    scores=np.asarray(history.val_auc)
    top_epochs=scores.argsort()[-top:][::-1]
    print(scores[top_epochs])
    os.system('mkdir best_weights')

    for i in range(top):
        weights_path='checkpoints_fold{}/epoch{}.ckpt'.format(fold,history.epoch[top_epochs[i]])
        print(weights_path)
        os.system('cp {} best_weights/fold{}top{}.ckpt'.format(weights_path,fold,i+1))


for i in range(1):
    get_best_weights_from_fold(i)
