import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

df=pd.read_csv("kmer_test.csv")


labels = df.name
scores = df.score
score_no_attention = df.score_no_attention

x = np.arange(len(labels))  # the label locations
width = 0.35
x1=x-width/2
x2=x+width/2
x1[0]=0
# the width of the bars
rects1_width=np.ones(len(x))*width
rects1_width[0]=0.7
rects2_width=np.ones(len(x))*width
rects2_width[0]=0




font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 10}
fig, ax = plt.subplots()
rects1 = ax.bar(x1, scores, rects1_width,label='with transformer encoder',color='m')
rects2 = ax.bar(x2, score_no_attention, rects2_width,label='without transformer encoder',color='c')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Cross validation accuracy')
#ax.set_title('Scores by group and gender')
plt.ylim(0.75,0.925)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
# rects1.set_color('b')
# rects2.set_color('r')

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:05.3f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
#autolabel(rects2)

fig.tight_layout()
ax.set_aspect(30)
# plt.show()
# exit()
plt.savefig("../graphics/kmer_test.eps")
plt.clf()
plt.close()
