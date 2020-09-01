import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import pandas as pd

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)

df=pd.read_csv("nlayer_test.csv")


# fig, ax = plt.subplots()
# #rects1 = ax.bar(x, scores, width)
# rects1 = ax.plot(df.nlayer, df.acc, 'm--*',label='With transformer encoder')
# rects2 = ax.plot(df.nlayer, np.ones(len(df.nlayer))*0.868, 'b',label='without transformer encoder')
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_xlabel('Number of transformer encoder layers')
# ax.set_ylabel('Cross validation accuracy')
# #ax.set_title('Scores by group and gender')
# plt.ylim(0.865,0.89)
# ax.set_xticks(df.nlayer)
# #ax.set_xticklabels(labels[1:])
# ax.legend()
# for line in ax.get_lines():
#     line.set_linewidth(4)


plt.plot(df.nlayer,df.acc,'m--o',linewidth=4,markersize=12)
#plt.title("5 fold accuracy vs number of transformer layer")
plt.ylabel("Cross validation accuracy")
plt.xlabel("Number of transformer encoder layers")
# for line in ax.get_lines():
#     line.set_linewidth(4)
#plt.ylim(0.85,0.925)
# plt.show()
# exit()
plt.tight_layout()
plt.savefig("../graphics/layers_acc.eps")
plt.show()
plt.clf()
#plt.show()
