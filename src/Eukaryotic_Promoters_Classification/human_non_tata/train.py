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
    parser.add_argument('--workers', type=int, default=4, help='number of workers')
    opts = parser.parse_args()

#     if len(opts.kmers)==1:

#         opts.kmers=[opts.kmers]
    return opts


def train_fold():

    opts=get_args()
    os.system('mkdir logs')
    seed_everything(20)
    #gpu selection
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df=pd.read_csv(opts.path)

    sequences=np.asarray(df.sequence)
    labels=np.asarray(df.label)

    train_indices, val_indices, test_indices=iter_split_strict(sequences,labels,opts.fold)
    dataset=PromoterDataset(sequences[train_indices],labels[train_indices])
    val_dataset=PromoterDataset(sequences[val_indices],labels[val_indices])
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=opts.batch_size,shuffle=True,num_workers=opts.workers)
    val_dataloader=torch.utils.data.DataLoader(val_dataset,batch_size=opts.batch_size*2,shuffle=False)

    #checkpointing
    checkpoints_folder='checkpoints_fold{}'.format((opts.fold))
    csv_file='logs/log_fold{}.csv'.format((opts.fold))
    columns=['epoch','train_loss','train_acc',
             'val_loss','val_acc','val_precision','val_recall','val_f1','val_mcc']
    logger=CSVLogger(columns,csv_file)

    #build model and logger
    model=NucleicTransformer(opts.ntoken, opts.nclass, opts.ninp, opts.nhead, opts.nhid,
                           opts.nlayers, opts.kmer_aggregation, kmers=opts.kmers,
                           dropout=opts.dropout).to(device)
    optimizer=torch.optim.Adam(model.parameters(), weight_decay=opts.weight_decay)
    criterion=nn.CrossEntropyLoss(reduction='none')
    lr_schedule=lr_AIAYN(optimizer,opts.ninp,opts.warmup_steps,opts.lr_scale)
    # Initialization
#     opt_level = 'O1'
#     model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    model = nn.DataParallel(model)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total number of paramters: {}'.format(pytorch_total_params))

    print("Starting training for fold {}/{}".format(opts.fold,opts.nfolds))
    #training loop
    for epoch in range(opts.epochs):
        model.train(True)
        t=time.time()
        total_loss=0
        #dataset.switch_mode(training=True)
        #dataset.update_batchsize(opts.batch_size)
        optimizer.zero_grad()
        train_preds=[]
        recon_preds=[]
        true_seqs=[]
        #step=0
        total_steps=len(dataloader)
        ground_truths=[]
        for step,data in enumerate(dataloader):
            lr=lr_schedule.step()
            #data=dataset[step]
            src=data['data'].long()
            #directions=data['directions']
            #directions=directions.reshape(len(directions),1)*np.ones(src.shape)
            #src=src.to(device).long()
            labels=data['labels'].to(device).long()
            mutated_sequence=mutate_dna_sequence(src,opts.nmute).to(device).long()
            output=model(mutated_sequence)
            #print(attention_weights.shape)
            loss=torch.mean(criterion(output,labels))#+\
            # 0.5*torch.mean(criterion(error_sequence[:,:81].reshape(-1,2),error_mask.reshape(-1).long()))+\
            # 0.5*torch.mean(criterion(recon_sequence[:,:81].reshape(-1,4),src.reshape(-1)))

            loss.backward()
#             with amp.scale_loss(loss, optimizer) as scaled_loss:
#                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            total_loss+=loss
            predictions = torch.argmax(output,dim=1).squeeze().cpu().numpy()
            train_preds.append(predictions)
            ground_truths.append(labels.cpu().numpy())
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.3f} Lr:{:.6f} Time: {:.1f}"
                           .format(epoch+1, opts.epochs, step+1, total_steps, total_loss/(step+1) , lr,time.time()-t),end='\r',flush=True) #total_loss/(step+1)
            #break
        print('')
        train_preds=np.concatenate(train_preds)
        ground_truths=np.concatenate(ground_truths)
        #ground_truths=dataset.labels
        train_acc=Metrics.accuracy(train_preds,ground_truths)
        train_loss=total_loss/(step+1)

        val_loss,val_acc,val_precision,val_recall,val_f1,val_mcc=validate(model,device,val_dataloader,batch_size=opts.batch_size*2)
        print("Epoch {} train acc: {}".format(epoch+1,train_acc))

        to_log=[epoch+1,train_loss,train_acc,val_loss,val_acc,val_precision,val_recall,val_f1,val_mcc]
        logger.log(to_log)


        if (epoch+1)%opts.save_freq==0:
            save_weights(model,optimizer,epoch,checkpoints_folder)


    get_best_weights_from_fold(opts.fold)

train_fold()


# for i in range(3,10):
    # ngrams=np.arange(2,i)
    # print(ngrams)
    # train_fold(0,ngrams)
# # train_fold(0,[2,3,4])
