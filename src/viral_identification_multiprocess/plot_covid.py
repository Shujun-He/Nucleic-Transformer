import pickle
import numpy as np
from Functions import *
from Dataset import *

nts=[
"A",
"T",
"G",
"C"]

with open("covid_results.p",'rb') as f:
    labels,preds=pickle.load(f)
preds_map=np.ones((99,300))*preds[:,-1].reshape(99,1)

with open('../../covid_genome.txt') as f:
    sequence=''
    for line in f:
        sequence+=line.strip()
sequence=nucleatide2int(sequence)
seqs=sequence[:len(sequence)//300*300].reshape(-1,300)
#seqs=sequence[:len(sequence)//300*300].reshape(-1,300)


import matplotlib
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 1}

matplotlib.rc('font', **font)

fig, ax = plt.subplots()
im = ax.matshow(preds_map,cmap='inferno')
cbar=fig.colorbar(im,fraction=0.05)
cbar.ax.tick_params(labelsize=10)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(preds_map)):
    for j in range(len(preds_map[0,:])):
        text = ax.text(j, i, nts[seqs[i, j]],color="g")
                       #ha="center", va="center", )
ax.set_aspect(3)
fig.tight_layout()
#plt.show()
#plt.savefig('covid_genome_scanner.png',dpi=1000)
plt.savefig('covid_genome_scanner.pdf')
plt.clf()
#
