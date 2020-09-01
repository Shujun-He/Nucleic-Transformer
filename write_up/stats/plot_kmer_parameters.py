import matplotlib.pyplot as plt
import matplotlib
import numpy as np

d_model=512
kmers=np.arange(11)

convs=[]
embeddings=[]
for k in kmers:
    convs.append(d_model**2*(k+1))
    embeddings.append(d_model*4**k)

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)
fig, ax = plt.subplots()
#rects1 = ax.bar(x, scores, width)
rects1 = ax.plot(kmers+1,np.log(convs), 'b--*',label='K-mer representation by convolutions',linewidth=4,markersize=12)
rects2 = ax.plot(kmers+1,np.log(embeddings), 'r--o',label='K-mer representation by embeddings',linewidth=4,markersize=12)
# rects1 = ax.plot(kmers+1,convs, 'b--*',label='K-mer representation by convolutions')
# rects2 = ax.plot(kmers+1,embeddings, 'r--o',label='K-mer representation by embeddings')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('k')
ax.set_ylabel('Log(number of parameters)')
#ax.set_aspect(0.4)
#ax.set_title('Scores by group and gender')
ax.legend()
fig.tight_layout()
#plt.margins(0,0)
plt.savefig('../graphics/kmer_parameter_complexity.eps')

plt.show()
