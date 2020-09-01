import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['end2end','2-model combined','3-model combined']
viraminer=[0.897,0.923,0]
nttransformer=[0.927,0.933,0.934]

# labels = ['Viraminer end2end','Nucleic Transformer end2end',
#          'Viraminer finetuned','Nucleic Transformer 2-model','Nucleic Transformer 3-model']
# results=[0.897,0.923,0.921,0.930,0.934]


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, viraminer, width, label='Viraminer',color='c')
rects2 = ax.bar(x + width/2, nttransformer, width, label='Nucleic Transformer',color='m')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('AUC')
plt.ylim(0.8,0.96)
ax.set_title('Viraminer vs Nucleic Transformer')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper left')


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:04.3f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')



autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig("../graphics/viral_results.eps")
plt.show()
