import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import pandas as pd

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)

df=pd.read_csv("dmodel_test.csv")



fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('$d_{model}$')
ax1.set_ylabel('Test AUC', color='m')
ax1.plot(df.dmodel,df.auc,'m--o',linewidth=4,markersize=12)
ax1.tick_params(axis='y', labelcolor='m')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Number of parameters (millions)', color='c')  # we already handled the x-label with ax1
ax2.plot(df.dmodel,df.nparam/1e6,'c--o',linewidth=4,markersize=12)
ax2.tick_params(axis='y', labelcolor='c')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig("../graphics/dmodel_test.eps")


plt.show()
