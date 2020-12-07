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

# json_path=os.path.join(opts.path,'train.json')
# data,labels=get_data(json_path)
# exit()


#checkpointing
checkpoints_folder='checkpoints_fold{}'.format((opts.fold))
csv_file='log_fold{}.csv'.format((opts.fold))
columns=['epoch','train_loss','train_acc','recon_acc',
         'val_loss','val_auc','val_acc','val_sens','val_spec']
#logger=CSVLogger(columns,csv_file)

#build model and logger
fold_models=[]
folds=np.arange(opts.nfolds)
for fold in folds:
    MODELS=[]
    for i in range(5):

        model=NucleicTransformer(opts.ntoken, opts.nclass, opts.ninp, opts.nhead, opts.nhid,
                               opts.nlayers, opts.kmer_aggregation, kmers=opts.kmers,
                               dropout=opts.dropout).to(device)
        optimizer=torch.optim.Adam(model.parameters(), weight_decay=opts.weight_decay)
        criterion=nn.CrossEntropyLoss(reduction='none')
        lr_schedule=lr_AIAYN(optimizer,opts.ninp,opts.warmup_steps,opts.lr_scale)
        # Initialization
        opt_level = 'O1'
        #model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
        model = nn.DataParallel(model)


        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('Total number of paramters: {}'.format(pytorch_total_params))

        model.load_state_dict(torch.load("best_weights/fold{}top{}.ckpt".format(fold,i+1)))
        #model.load_state_dict(torch.load("checkpoints_fold0/epoch{}.ckpt".format(i)))
        model.eval()
        MODELS.append(model)

    dict=MODELS[0].module.state_dict()
    for key in dict:
        for i in range(1,len(MODELS)):
            dict[key]=dict[key]+MODELS[i].module.state_dict()[key]

        dict[key]=dict[key]/float(len(MODELS))

    MODELS[0].module.load_state_dict(dict)
    avg_model=MODELS[0]
    fold_models.append(avg_model)



def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):
    token2int = {x:i for i, x in enumerate('ACGU().BEHIMSX')}
    return np.transpose(
        np.array(
            df[cols]
            .applymap(lambda seq: [token2int[x] for x in seq])
            .values
            .tolist()
        ),
        (0, 2, 1)
    )

alt_structure_df=pd.read_csv(os.path.join(opts.path,'test_alternative_structure_loops.csv'))
alt_structure_df_50C=pd.read_csv(os.path.join(opts.path,'test_alternative_structure_loops_50C.csv'))

#alt_input=

json_path=os.path.join(opts.path,'test.json')
test = pd.read_json(json_path, lines=True)
aug_df_path=os.path.join(opts.path,'aug_data1.csv')
aug_df=pd.read_csv(aug_df_path)
aug_test,indices=aug_data(test,aug_df)
aug_test=aug_test.loc[indices]
#aug_test=test
#dataloader
ls_indices=test.seq_length==130
ls_indices2=aug_test.seq_length==130
long_data=test[ls_indices]
data=preprocess_inputs(test[ls_indices])
data=data.reshape(1,*data.shape)
alt_data=get_alt_structures(alt_structure_df[ls_indices])
alt_data2=get_alt_structures_50C(alt_structure_df_50C[ls_indices])
data=np.concatenate([data,alt_data,alt_data2],0).transpose(1,0,2,3)


# data2=preprocess_inputs(aug_test[ls_indices2])
# data=np.stack([data,data2],axis=0)

ids=np.asarray(long_data.id.to_list())
long_dataset=RNADataset(long_data.sequence.to_list(),np.zeros(len(ls_indices)),ids, np.arange(len(ls_indices)),opts.path,training=False)
long_dataloader=DataLoader(long_dataset, batch_size=opts.batch_size,
                        shuffle=False)


ss_indices=test.seq_length==107
ss_indices2=aug_test.seq_length==107
short_data=test[ss_indices]
ids=np.asarray(short_data.id.to_list())
data=preprocess_inputs(test[ss_indices])
data=data.reshape(1,*data.shape)
alt_data=get_alt_structures(alt_structure_df[ss_indices])
alt_data2=get_alt_structures_50C(alt_structure_df_50C[ss_indices])
data=np.concatenate([data,alt_data,alt_data2],0).transpose(1,0,2,3)
# print(data.shape)
# exit()
short_dataset=RNADataset(short_data.sequence.to_list(),np.zeros(len(ss_indices)), ids, np.arange(len(ss_indices)),opts.path,training=False)
short_dataloader=DataLoader(short_dataset, batch_size=opts.batch_size,
                        shuffle=False)


# distance_mask1=get_distance_mask(130)
# distance_mask1=torch.tensor(distance_mask1).float().to(device).reshape(1,130,130)
# distance_mask2=get_distance_mask(107)
# distance_mask2=torch.tensor(distance_mask2).float().to(device).reshape(1,107,107)
#
#
# distance_mask1=get_distance_mask(130)
# distance_mask1=torch.tensor(distance_mask1).float().to(device).reshape(1,130,130)
# distance_mask2=get_distance_mask(107)
# distance_mask2=torch.tensor(distance_mask2).float().to(device).reshape(1,107,107)
#exit()

nts='ACGU().BEHIMSX'
ids=[]
preds=[]
with torch.no_grad():
    for batch in tqdm(long_dataloader):
        sequence=batch['data'].to(device)
        bpps=batch['bpp'].float().to(device)
        # print(bpps.shape)
        # print(sequence.shape)
        # exit()
        avg_preds=[]
        outputs=[]
        for i in range(sequence.shape[1]):
            temp=[]
            for model in fold_models:
                #outputs.append(model(sequence[:,i],bpps[:,i]))
                temp.append(model(sequence[:,i],bpps[:,i]))

            temp=torch.stack(temp,0).mean(0)
            outputs.append(temp)

        outputs=torch.stack(outputs,1).cpu().numpy()#.mean(0)
        #avg_preds=outputs.cpu().numpy()
            # avg_preds.append(output.cpu().numpy())
        #avg_preds=np.mean(avg_preds,axis=0)
        for pred in outputs:
            preds.append(pred)
        for string in batch['id']:
            ids.append(string)

    for batch in tqdm(short_dataloader):
        sequence=batch['data'].to(device)
        bpps=batch['bpp'].float().to(device)

        avg_preds=[]
        outputs=[]
        for i in range(sequence.shape[1]):
            temp=[]
            for model in fold_models:
                #outputs.append(model(sequence[:,i],bpps[:,i]))
                temp.append(model(sequence[:,i],bpps[:,i]))

            temp=torch.stack(temp,0).mean(0)
            outputs.append(temp)

        outputs=torch.stack(outputs,1).cpu().numpy()#.mean(0)
        #avg_preds=outputs.cpu().numpy()
        for pred in outputs:
            preds.append(pred)
        for string in batch['id']:
            ids.append(string)
#exit()


preds_to_csv=[[] for i in range(len(test))]
test_ids=test.id.to_list()
for i in tqdm(range(len(preds))):
    index=test_ids.index(ids[i])
    preds_to_csv[index].append(preds[i])

to_csv=[]
for i in tqdm(range(len(preds_to_csv))):
    to_write=np.asarray(preds_to_csv[i][0])
    to_write=to_write.transpose(1,0,2)
    for vector in to_write:
        to_csv.append(vector)
to_csv=np.asarray(to_csv)

# avail_packages=['vienna_2', 'nupack', 'contrafold_2', 'eternafold', 'rnastructure','rna_soft']
avail_packages=['contrafold_2', 'eternafold', 'nupack', 'rnastructure', 'vienna_2', 'rnasoft']
submission=pd.read_csv(os.path.join(opts.path,'sample_submission.csv'))

to_csv=np.concatenate([to_csv[:,:5],to_csv[:,6:11],to_csv[:,6].reshape(to_csv.shape[0],1,-1),to_csv[:,11].reshape(to_csv.shape[0],1,-1)],1)

for i, pkg in enumerate(avail_packages):
    pkg_predictions=np.stack([to_csv[:,i*2],to_csv[:,i*2+1]],0).mean(0)
    pkg_sub=submission.copy()
    pkg_sub.iloc[:,1:]=pkg_predictions
    pkg_sub.to_csv(f"{pkg}.csv",index=False)
    #os.system(f'kaggle competitions submit -c stanford-covid-vaccine -f {pkg}.csv -m "Message"')



with open('predictions.p','wb+') as f:
    pickle.dump(preds,f)


submission.iloc[:,1:]=to_csv.mean(1)

submission.to_csv('submission_new_squared.csv',index=False)
