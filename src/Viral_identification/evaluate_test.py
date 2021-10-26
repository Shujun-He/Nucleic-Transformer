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
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0,1',  help='which gpu to use')
    parser.add_argument('--path', type=str, default='../', help='path of csv file with DNA sequences and labels')
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
    parser.add_argument('--val_freq', type=int, default=1, help='which fold to train')
    opts = parser.parse_args()
    return opts


opts=get_args()
#gpu selection
os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#lr=0

#checkpointing
checkpoints_folder='checkpoints_fold{}'.format((opts.fold))
csv_file='log_fold{}.csv'.format((opts.fold))
columns=['epoch','train_loss','train_acc','recon_acc',
         'val_loss','val_auc','val_acc','val_sens','val_spec']
#logger=CSVLogger(columns,csv_file)

#build model and logger
MODELS=[]
for i in range(3):
    model=NucleicTransformer(opts.ntoken, opts.nclass, opts.ninp, opts.nhead, opts.nhid,
                           opts.nlayers, opts.kmer_aggregation, kmers=opts.kmers,
                           dropout=opts.dropout).to(device)
    optimizer=torch.optim.Adam(model.parameters(), weight_decay=opts.weight_decay)
    criterion=nn.CrossEntropyLoss(reduction='none')
    lr_schedule=lr_AIAYN(optimizer,opts.ninp,opts.warmup_steps,opts.lr_scale)
    # Initialization
    opt_level = 'O1'
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    model = nn.DataParallel(model)


    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total number of paramters: {}'.format(pytorch_total_params))

    model.load_state_dict(torch.load("best_weights/fold0top{}.ckpt".format(i+1)))
    model.eval()
    MODELS.append(model)

dict=MODELS[0].module.state_dict()
for key in dict:
    for i in range(1,len(MODELS)):
        dict[key]=dict[key]+MODELS[i].module.state_dict()[key]

    dict[key]=dict[key]/float(len(MODELS))

MODELS[0].module.load_state_dict(dict)
avg_model=MODELS[0]

def geometric_mean(preds):
    gmean=np.ones(preds.shape[1:])

    for pred in preds:
        gmean=gmean*pred

    gmean=gmean**(1/len(preds))
    return gmean

df=pd.read_csv('../fullset_test.csv',header=None)

seqs=[]
labels=[]

for i in range(len(df)):
    seqs.append(nucleatide2int(df.iloc[i,1]))
    labels.append(df.iloc[i,2])
labels=np.asarray(labels).astype("int")
seqs=np.asarray(seqs).astype("int")


batch_size=128
batches=np.around(len(df)/batch_size+0.5).astype('int')
preds=[]
softmax = nn.Softmax(dim=1)
for i in tqdm(range(batches)):
    with torch.no_grad():
        outputs=[]
        #for model in MODELS:
        x=torch.Tensor(seqs[i*batch_size:(i+1)*batch_size]).to(device).long()
        y=softmax(avg_model(x))
        #outputs.append(softmax(y).cpu().numpy())
        for vec in y:
            preds.append(vec.cpu().numpy())

from sklearn import metrics
preds=np.asarray(preds)
auc=metrics.roc_auc_score(labels,preds[:,1])

with open("test_results.p",'wb+') as f:
    pickle.dump([labels,preds],f)


print(auc)
with open("test_score.txt",'w+') as f:
    f.write("test auc score: {}".format(auc))




# for i in range(3,10):
    # ngrams=np.arange(2,i)
    # print(ngrams)
    # train_fold(0,ngrams)
# # train_fold(0,[2,3,4])
