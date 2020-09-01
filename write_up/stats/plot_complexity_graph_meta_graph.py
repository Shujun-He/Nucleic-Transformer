import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}

L=81
kmers=6
ks=np.arange(kmers)

graph_comp=[]
metagraph_comp=[]

for i in ks:
    graph_comp.append((L-i)**2)
    metagraph_comp.append(((2*L-i)*(i+1)/2)**2)

graph_comp=np.asarray(graph_comp)
metagraph_comp=np.asarray(metagraph_comp)

# plt.plot(ks+1,np.log(graph_comp))
# plt.plot(ks+1,np.log(metagraph_comp))
# plt.show()


matplotlib.rc('font', **font)
fig, ax = plt.subplots()
#rects1 = ax.bar(x, scores, width)
rects1 = ax.plot(ks+1,np.log(graph_comp), 'b--*',label='K-mer graph',linewidth=4,markersize=12)
rects2 = ax.plot(ks+1,np.log(metagraph_comp), 'r--o',label='K-mer meta-graph',linewidth=4,markersize=12)
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('k')
ax.set_ylabel('Log(computational complexity)')
#ax.set_title('Scores by group and gender')
ax.legend()
fig.tight_layout()
plt.savefig('../graphics/graph_metagraph_complexity.eps')

plt.show()
