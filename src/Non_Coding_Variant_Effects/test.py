import os
import torch
import torch.nn as nn
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
import scipy.io
import h5py
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0',  help='which gpu to use')
    parser.add_argument('--path', type=str, default='../', help='path of csv file with DNA sequences and labels')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=24, help='size of each batch during training')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight dacay used in optimizer')
    parser.add_argument('--ntoken', type=int, default=4, help='number of tokens to represent DNA nucleotides (should always be 4)')
    parser.add_argument('--nclass', type=int, default=919, help='number of classes from the linear decoder')
    parser.add_argument('--ninp', type=int, default=512, help='ninp for transformer encoder')
    parser.add_argument('--nhead', type=int, default=8, help='nhead for transformer encoder')
    parser.add_argument('--nhid', type=int, default=2048, help='nhid for transformer encoder')
    parser.add_argument('--nlayers', type=int, default=6, help='nlayers for transformer encoder')
    parser.add_argument('--save_freq', type=int, default=1, help='saving checkpoints per save_freq epochs')
    parser.add_argument('--dropout', type=float, default=.1, help='transformer dropout')
    parser.add_argument('--warmup_steps', type=int, default=3200, help='training schedule warmup steps')
    parser.add_argument('--lr_scale', type=float, default=0.1, help='learning rate scale')
    parser.add_argument('--nmute', type=int, default=18, help='number of mutations during training')
    parser.add_argument('--kmers', type=int, nargs='+', default=[7], help='k-mers to be aggregated')
    #parser.add_argument('--kmer_aggregation', type=bool, default=True, help='k-mers to be aggregated')
    parser.add_argument('--kmer_aggregation', dest='kmer_aggregation', action='store_true')
    parser.add_argument('--no_kmer_aggregation', dest='kmer_aggregation', action='store_false')
    parser.set_defaults(kmer_aggregation=True)
    parser.add_argument('--nfolds', type=int, default=5, help='number of cross validation folds')
    parser.add_argument('--fold', type=int, default=0, help='which fold to train')
    parser.add_argument('--val_freq', type=int, default=1, help='which fold to train')
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers')
    opts = parser.parse_args()
    return opts

#def train_fold():

opts=get_args()
seed_everything(2020)
#gpu selection
os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# import pickle
# with open('../DeepSea_TrainVal.p','rb') as f:
#     train_seqs,train_labels,val_seqs,val_labels=pickle.load(f)

test_data = scipy.io.loadmat('../deepsea_train/test.mat')
test_seqs = np.array(test_data['testxdata']).astype('uint8')
test_labels = np.array(test_data['testdata']).astype('uint8')

#exit()

dataset=DeepSeaDataset(test_seqs,test_labels)
loader=torch.utils.data.DataLoader(dataset,batch_size=opts.batch_size*4,shuffle=False,num_workers=opts.num_workers)

#exit()
#lr=0

#checkpointing
checkpoints_folder='checkpoints_fold{}'.format((opts.fold))
log=pd.read_csv('log_fold{}.csv'.format((opts.fold)))
best_epoch=log['epoch'][log.val_loss.argmin()]
#last_epoch=log['epoch'].iloc[-1]
#print(f"best epoch is {best_epoch}")
best_weights=torch.load(f"checkpoints_fold0/epoch{best_epoch}.ckpt")

#exit()

#build model and logger
MODELS=[]
for i in range(1):
    model=NucleicTransformer(opts.ntoken, opts.nclass, opts.ninp, opts.nhead, opts.nhid,
                           opts.nlayers, opts.kmer_aggregation, kmers=opts.kmers,
                           dropout=opts.dropout).to(device)
    optimizer=torch.optim.Adam(model.parameters(), weight_decay=opts.weight_decay)
    criterion=nn.BCEWithLogitsLoss()
    lr_schedule=lr_AIAYN(optimizer,opts.ninp,opts.warmup_steps,opts.lr_scale)
    # Initialization
    opt_level = 'O1'
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    model = nn.DataParallel(model)
    #exit()
    epoch=log['epoch'][log.val_loss.argsort()[i]]
    #last_epoch=log['epoch'].iloc[-1]
    print(f"best epoch is {epoch}")
    best_weights=torch.load(f"checkpoints_fold0/epoch{epoch}.ckpt")
    model.load_state_dict(best_weights)
    MODELS.append(model)


#softmax = nn.Softmax(dim=1)

pytorch_total_params = sum(p.numel() for p in model.parameters())
print('Total number of paramters: {}'.format(pytorch_total_params))

#print("Starting training for fold {}/{}".format(opts.fold,opts.nfolds))
batches=len(loader)
model.train(False)
total=0
predictions=[]
outputs=[]
ground_truths=[]
loss=0
criterion=nn.BCEWithLogitsLoss()
with torch.no_grad():

    for data in tqdm(loader):

        X=data['data'].to(device).long()
        Y=data['labels'].to(device).float()

        temp=[]
        for model in MODELS:
            output= torch.sigmoid(model(X))+\
            torch.sigmoid(model(get_complementary_sequence_deepsea(X)))#+\
            # torch.sigmoid(model(X[:,4:]))+\
            # torch.sigmoid(model(get_complementary_sequence_deepsea(X[:,4:])))
            # torch.sigmoid(model(X[:,1:]))+\
            # torch.sigmoid(model(get_complementary_sequence_deepsea(X[:,1:])))+\

            probs = output/2
            temp.append(probs)
        probs=torch.stack(temp,0).mean(0)
        del X
        loss+=criterion(output,Y)
        #probs = output)
        for pred in probs:
            predictions.append(pred.cpu().numpy()>0.5)
        for vector in probs:
            outputs.append(vector.cpu().numpy())
        for t in Y:
            ground_truths.append(t.cpu().numpy())
        del output
torch.cuda.empty_cache()
val_loss=(loss/batches).cpu()
ground_truths=np.asarray(ground_truths)#.reshape(-1)
predictions=np.asarray(predictions)#.reshape(-1)
outputs=np.asarray(outputs)#.reshape(-1)
#score=metrics.cohen_kappa_score(ground_truths,predictions,weights='quadratic')
# val_acc=Metrics.accuracy(predictions,ground_truths)
# auc=metrics.roc_auc_score(ground_truths,outputs)
# val_sens=Metrics.sensitivity(predictions,ground_truths)
# val_spec=Metrics.specificity(predictions,ground_truths)

print(val_loss)

with open('test_results.p','wb+') as f:
    pickle.dump([outputs,ground_truths],f)

#train_fold()
