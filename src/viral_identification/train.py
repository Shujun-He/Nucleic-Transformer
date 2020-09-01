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
    parser.add_argument('--stride', type=int, default=1, help='stride used in k-mer convolution')
    opts = parser.parse_args()
    return opts

def train_fold():

    opts=get_args()
    #gpu selection
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset=ViraMiner_Dataset(opts.path,opts.fold,opts.nfolds,batch_size=opts.batch_size)
    #lr=0

    #checkpointing
    checkpoints_folder='checkpoints_fold{}'.format((opts.fold))
    csv_file='log_fold{}.csv'.format((opts.fold))
    columns=['epoch','train_loss','train_acc',
             'val_loss','val_auc','val_acc','val_sens','val_spec']
    logger=CSVLogger(columns,csv_file)

    #build model and logger
    model=NucleicTransformer(opts.ntoken, opts.nclass, opts.ninp, opts.nhead, opts.nhid,
                           opts.nlayers, opts.kmer_aggregation, kmers=opts.kmers, stride=opts.stride,
                           dropout=opts.dropout).to(device)
    optimizer=torch.optim.Adam(model.parameters(), weight_decay=opts.weight_decay)
    criterion=nn.CrossEntropyLoss(reduction='none')
    lr_schedule=lr_AIAYN(optimizer,opts.ninp,opts.warmup_steps,opts.lr_scale)
    # Initialization
    opt_level = 'O1'
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    model = nn.DataParallel(model)
    softmax = nn.Softmax(dim=1)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total number of paramters: {}'.format(pytorch_total_params))

    print("Starting training for fold {}/{}".format(opts.fold,opts.nfolds))
    #training loop
    for epoch in range(opts.epochs):
        model.train(True)
        t=time.time()
        total_loss=0
        dataset.switch_mode(training=True)
        dataset.update_batchsize(opts.batch_size)
        optimizer.zero_grad()
        train_preds=[]
        #recon_preds=[]
        #true_seqs=[]
        for step in range(len(dataset)):
        #for step in range(1):
            lr=lr_schedule.step()
            data=dataset[step]
            src=data['data']
            #directions=data['directions']
            #directions=directions.reshape(len(directions),1)*np.ones(src.shape)
            src=torch.Tensor(src.copy()).to(device).long()
            labels=torch.Tensor(data['labels']).to(device).long()
            #directions=torch.Tensor(directions).to(device).long()
            mutated_sequence=mutate_dna_sequence(data['data'],opts.nmute)
            #error_mask=torch.Tensor((mutated_sequence!=data['data']).reshape(-1)).to(device,dtype=torch.bool)
            mutated_sequence=torch.Tensor(mutated_sequence).to(device).long()
            output,attention_weights=model(mutated_sequence,None)
            #print(attention_weights.shape)
            loss_weight=torch.ones(len(output),device=device)
            #print(output.shape)
            #print(labels.shape)
            # pt=softmax(output)
            # loss_weight=(1-pt[np.arange(len(labels)),labels])**2
            #loss_weight[labels==1]=5
            loss=torch.mean(criterion(output,labels))#+\
            # 0.5*torch.mean(criterion(error_sequence[:,:81].reshape(-1,2),error_mask.reshape(-1).long()))+\
            # 0.5*torch.mean(criterion(recon_sequence[:,:81].reshape(-1,4),src.reshape(-1)))


            with amp.scale_loss(loss, optimizer) as scaled_loss:
               scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            total_loss+=loss
            predictions = torch.argmax(output,dim=1).squeeze().cpu().numpy()
            #recon_predictions = torch.argmax(recon_sequence[:,:81].reshape(-1,4),dim=1).squeeze().cpu().numpy()
            train_preds.append(predictions)
            #recon_preds.append(recon_predictions)
            #true_seqs.append(data['data'].reshape(-1))
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.3f} Lr:{:.6f} Time: {:.1f}"
                           .format(epoch+1, opts.epochs, step+1, dataset.train_batches, total_loss/(step+1) , lr,time.time()-t),end='\r',flush=True) #total_loss/(step+1)
        print('')
        train_preds=np.concatenate(train_preds)
        #recon_preds=np.concatenate(recon_preds)
        #true_seqs=np.concatenate(true_seqs)
        ground_truths=dataset.labels[dataset.train_indices]
        train_acc=Metrics.accuracy(train_preds,ground_truths)
        train_loss=total_loss/(step+1)
        #recon_acc=np.sum(recon_preds==true_seqs)/len(recon_preds)

        if (epoch+1)%opts.val_freq==0:
            val_loss,auc,val_acc,val_sens,val_spec=validate(model,device,dataset,batch_size=opts.batch_size)
            print("Epoch {} train acc: {}".format(epoch+1,train_acc))

            to_log=[epoch+1,train_loss,train_acc,val_loss,auc,val_acc,val_sens,val_spec]
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