from sklearn import metrics
import numpy as np



a=np.random.uniform(size=(1000))
b=np.ones(1000)
b[100]=0

auc=metrics.roc_auc_score(b,a)
