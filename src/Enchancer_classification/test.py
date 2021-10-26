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
import argparse

try:
    #from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
import pickle
#gpu selection
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from sklearn.metrics import matthews_corrcoef
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0',  help='which gpu to use')
    parser.add_argument('--path', type=str, default='../v9d3.csv', help='path of csv file with DNA sequences and labels')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=24, help='size of each batch during training')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight dacay used in optimizer')
    parser.add_argument('--ntoken', type=int, default=4, help='number of tokens to represent DNA nucleotides (should always be 4)')
    parser.add_argument('--nclass', type=int, default=2, help='number of classes from the linear decoder')
    parser.add_argument('--ninp', type=int, default=512, help='ninp for transformer encoder')
    parser.add_argument('--nhead', type=int, default=8, help='nhead for transformer encoder')
    parser.add_argument('--nhid', type=int, default=2048, help='nhid for transformer encoder')
    parser.add_argument('--nlayers', type=int, default=6, help='nlayers for transformer encoder')
    parser.add_argument('--save_freq', type=int, default=1, help='saving checkpoints per save_freq epochs')
    parser.add_argument('--dropout', type=float, default=.1, help='transformer dropout')
    parser.add_argument('--warmup_steps', type=int, default=3200, help='training schedule warmup steps')
    parser.add_argument('--lr_scale', type=float, default=0.1, help='learning rate scale')
    parser.add_argument('--nmute', type=int, default=18, help='number of mutations during training')
    parser.add_argument('--kmers', type=int, nargs='+', default=[2,3,4,5,6], help='k-mers to be aggregated')
    #parser.add_argument('--kmer_aggregation', type=bool, default=True, help='k-mers to be aggregated')
    parser.add_argument('--kmer_aggregation', dest='kmer_aggregation', action='store_true')
    parser.add_argument('--no_kmer_aggregation', dest='kmer_aggregation', action='store_false')
    parser.set_defaults(kmer_aggregation=True)
    parser.add_argument('--nfolds', type=int, default=5, help='number of cross validation folds')
    parser.add_argument('--fold', type=int, default=0, help='which fold to train')
    opts = parser.parse_args()
    return opts

def evaluate_fold(fold):

    #load data
    #opts=get_args()
    df=pd.read_csv(opts.path)

    sequences=np.asarray(df.sequence)
    labels=np.asarray(df.label)

    train_indices, val_indices=iter_split(sequences,labels,fold,opts.nfolds)
    # print(train_indices.shape)
    # print(val_indices.shape)
    # exit()
    test_dataset=PromoterDataset(sequences,labels)
    test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_size=opts.batch_size*2,shuffle=False)



    #init model
    model=NucleicTransformer(opts.ntoken, opts.nclass, opts.ninp, opts.nhead, opts.nhid,
                           opts.nlayers, opts.kmer_aggregation, kmers=opts.kmers,
                           dropout=opts.dropout,return_aw=True).to(device)
    #optimizer=torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=weight_decay)

    # Initialization
    # opt_level = 'O1'
    # model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    # print('Total number of paramters: {}'.format(pytorch_total_params))

    #evaluation loop
    #ground_truths=dataset.labels[dataset.val_indices]
    ensemble_predictions=[]
    acc=[]

    weights_path="best_weights/fold{}top1.ckpt".format(fold,i+1)
    print(weights_path)
    checkpoint=torch.load(weights_path)
    model.load_state_dict(checkpoint)
    predictions,attention_weights,sequences,ground_truths=predict(model,device,test_dataloader)
    # #validate(model,device,dataset,batch_size=batch_size*2)
    predictions=np.exp(predictions)/np.sum(np.exp(predictions),axis=1).reshape(len(predictions),1)
    ensemble_predictions.append(predictions)
    ensemble_predictions=np.asarray(ensemble_predictions)
    ensemble_predictions=np.mean(np.asarray(ensemble_predictions),axis=0)
    model.cpu()
    del model
    #del optimizer
    torch.cuda.empty_cache()
    return ensemble_predictions, ground_truths, attention_weights, sequences

opts=get_args()


predictions=[]
ground_truths=[]
attention_weights=[]
sequences=[]
for i in range(5):
    ngram=[7]
    p,t,at,seq= evaluate_fold(i)
    predictions.append(p)
    ground_truths.append(t)
    #print(at.shape)
    #attention_weights.append(at)
    #print(seq.shape)
    #sequences.append(seq)


probs=np.stack(predictions,0).mean(0)
ground_truths=np.stack(ground_truths,0).mean(0)
predictions=np.argmax(probs,axis=1)
#attention_weights=np.squeeze(np.concatenate(attention_weights,0)).astype('float16')
#sequences=np.asarray(sequences).reshape(-1,81)
acc=Metrics.accuracy(predictions,ground_truths)
sens=Metrics.sensitivity(predictions,ground_truths)
spec=Metrics.specificity(predictions,ground_truths)
MCC=matthews_corrcoef(ground_truths,predictions)

# prediction_dict={'predictions':np.squeeze(predictions),
#                  'ground_truths':np.squeeze(ground_truths),
#                  'attention_weights':np.squeeze(attention_weights),
#                  'sequences':np.squeeze(sequences.reshape(-1,81))
# }

# with open("prediction_dict.p","wb+") as f:
#     pickle.dump(prediction_dict,f)


with open("test_results.txt",'w+') as f:
    f.write(f"ACC: {acc}\n")
    f.write(f"sensitivity: {sens}\n")
    f.write(f"spec: {spec}\n")
    f.write(f"MCC: {MCC}\n")
