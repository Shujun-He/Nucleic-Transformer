import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
df=pd.read_csv("nmute_test.csv")


labels = df.nmute
scores = df.score

x = np.arange(len(labels)-1)  # the label locations
width = 0.7  # the width of the bars


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)
fig, ax = plt.subplots()
#rects1 = ax.bar(x, scores, width)
rects1 = ax.plot(x, scores[1:], 'm--*',label='with random mutations',markersize=12)
rects2 = ax.plot(x, np.ones(len(x))*scores[0], 'b',label='without random mutations')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('$n_{mute}$')
ax.set_ylabel('Cross validation accuracy')
#ax.set_title('Scores by group and gender')
plt.ylim(0.85,0.9)
ax.set_xticks(x)
ax.set_xticklabels(labels[1:])
ax.legend()
for line in ax.get_lines():
    line.set_linewidth(4)
#rects1[0].set_color('r')

# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{:05.3f}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')


#autolabel(rects1)

fig.tight_layout()
#
# exit()
plt.savefig("../graphics/nmute_test.eps")
plt.show()
plt.clf()
plt.close()
