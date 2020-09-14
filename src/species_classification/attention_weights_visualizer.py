import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable



def plot_attention_weights(attention, sentence, result, save_file=None,show=None,dpi=500):
    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(1, 1, 1)
    # plot the attention weights
    im=ax.matshow(attention, cmap='plasma',)
    #ax.matshow(attention, cmap='plasma',)
    #fig.colorbar(cax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    fontdict = {'fontsize': 8}

    ax.set_xticks(range(len(sentence)+2))
    ax.set_yticks(range(len(result)))

    ax.set_ylim(len(result)-1.5, -0.5)

    ax.set_xticklabels(
        [nt for nt in sentence],
        fontdict=fontdict)

    ax.set_yticklabels([nt for nt in sentence],
                       fontdict=fontdict)


    plt.tight_layout()
    if save_file:
        plt.savefig(save_file,dpi=dpi)
    if show:
        plt.show()
    plt.clf()
    plt.close()


def k_mer_aggregated2nxn(attention_weights,k_mers=[1,3,5,7,9],sequence_lenth=81):
    new_attention_weight_matrix=np.zeros(shape=(sequence_lenth,sequence_lenth))
    #new_attention_weight_matrix=attention_weights[:sequence_lenth,:sequence_lenth
    indices=[]
    for k in k_mers:
        k_mer_seq_length=sequence_lenth-k+1
        for j in range(k_mer_seq_length):
            indices.append([j+ii for ii in range(k)])

    for i in range(attention_weights.shape[0]):
        for j in range(attention_weights.shape[1]):
            total_slots=len(indices[i])*len(indices[j])
            value=attention_weights[i,j]#/total_slots
            for index_x in indices[i]:
                for index_y in indices[j]:
                    new_attention_weight_matrix[index_x,index_y]+=value

    return new_attention_weight_matrix
