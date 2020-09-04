import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from Functions import *
from Dataset import *
from Network import *
from LrScheduler import *
import Metrics
from Logger import CSVLogger
try:
    #from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
import pickle
#gpu selection
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def evaluate_fold(fold,ngrams,top):
    #hyperparamters
    epochs=150
    batch_size=24
    lr=1e-4
    weight_decay=0
    ntoken=4
    nclass=2
    ninp=512
    nhead=8
    nhid=2048
    nlayers=6
    save_freq=1
    dropout=0.1
    warmup_steps=1200
    lr_scale=0.1
    cos_annealing_epoch=-1
    min_lr=1e-7
    T=epochs-cos_annealing_epoch
    nmute=18
    nmask=15
    gap=25


    #load data
    path='../../data/v9d3.csv'
    nfolds=5
    dataset=PromoterDataset(path,fold,nfolds,batch_size=batch_size)

    #checkpointing
    checkpoints_folder='checkpoints_fold{}'.format((fold))
    csv_file='gap_log_fold{}.csv'.format((fold))
    columns=['epoch','gap_acc']
    logger=CSVLogger(columns,csv_file)

    #build model and logger


    #weights_path="checkpoints_fold{}/epoch{}.ckpt".format(fold,148)
    #weights_path="../rebuilt/checkpoints_fold{}/epoch{}.ckpt".format(fold,120)

    # opt_level = 'O1'
    # model, optimizer = amp.initialize(model,optimizer, opt_level=opt_level,verbosity=0)

    model=TransformerModel(ntoken, nclass, ninp, nhead, nhid, nlayers,ngrams=ngrams,dropout=dropout).to(device)
    optimizer=torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialization
    # opt_level = 'O1'
    # model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    # print('Total number of paramters: {}'.format(pytorch_total_params))

    #evaluation loop
    ground_truths=dataset.labels[dataset.val_indices]
    ensemble_predictions=[]
    acc=[]
    for i in range(top):

        weights_path="best_weights/fold{}top{}.ckpt".format(fold,i+1)
        print(weights_path)
        checkpoint=torch.load(weights_path)
        model.load_state_dict(checkpoint)
        predictions,attention_weights,sequences=predict(model,device,dataset)
        # #validate(model,device,dataset,batch_size=batch_size*2)
        predictions=np.exp(predictions)/np.sum(np.exp(predictions),axis=1).reshape(len(predictions),1)
        ensemble_predictions.append(predictions)
    ensemble_predictions=np.asarray(ensemble_predictions)
    ensemble_predictions=np.mean(np.asarray(ensemble_predictions),axis=0)
    model.cpu()
    del model
    del optimizer
    torch.cuda.empty_cache()
    return ensemble_predictions, ground_truths, attention_weights, sequences

predictions=[]
ground_truths=[]
attention_weights=[]
sequences=[]
for i in range(1):
    ngram=np.arange(2,7)
    p,t,at,seq= evaluate_fold(i,ngram,top=1)
    predictions.append(p)
    ground_truths.append(t)
    print(at.shape)
    attention_weights.append(at)
    sequences.append(seq)


predictions=np.concatenate(predictions)
ground_truths=np.concatenate(ground_truths)
predictions=np.argmax(predictions,axis=1)
attention_weights=np.squeeze(np.asarray(attention_weights))
sequences=np.asarray(sequences)
acc=Metrics.accuracy(predictions,ground_truths)

from attention_weights_visualizer import *

nxn_attention_weights=np.zeros(shape=(attention_weights.shape[0],attention_weights.shape[1],81,81))

for i in tqdm(range(nxn_attention_weights.shape[0])):
    for j in range(nxn_attention_weights.shape[1]):
        nxn_attention_weights[i,j]=k_mer_aggregated2nxn(attention_weights[i,j],np.arange(6)+1)

prediction_dict={'predictions':np.squeeze(predictions),
                 'ground_truths':np.squeeze(ground_truths),
                 'attention_weights':np.squeeze(nxn_attention_weights),
                 'sequences':np.squeeze(sequences)
}

with open("prediction_dict.p","wb+") as f:
    pickle.dump(prediction_dict,f)


with open("cv.txt",'w+') as f:
    f.write("5 fold cv acc:{}".format(acc))
