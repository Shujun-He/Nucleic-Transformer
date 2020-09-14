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
from torchvision import transforms, utils
from Mutation import *

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
    parser.add_argument('--viral_loss_weight', type=int, default=1, help='stride used in k-mer convolution')
    parser.add_argument('--workers', type=int, default=1, help='number of workers for dataloader')
    opts = parser.parse_args()
    return opts

def train_fold():
    #get arguments
    opts=get_args()

    #gpu selection
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #instantiate datasets
    df_path=os.path.join(opts.path,'fullset_train.csv')
    #dataset=ViraminerDataset(df_path,transform=transforms.Compose([Mutation(opts.nmute)]))
    seqs=get_seqs(os.path.join(opts.path,'100bp_seqs.p'))
    labels=get_labels(os.path.join(opts.path,'int_labels.p'))[:,3]
    keys,static_embeddings=get_embeddings(os.path.join(opts.path,'static_embeddings.p'))
    val_cutoff=int(0.8*len(seqs))
    test_cutoff=int(0.9*len(seqs))
    dataset=Seq2SpeciesDataset(seqs[:val_cutoff],labels[:val_cutoff])
    val_dataset=Seq2SpeciesDataset(seqs[val_cutoff:test_cutoff],labels[val_cutoff:test_cutoff])
    dataloader = DataLoader(dataset, batch_size=opts.batch_size,
                            shuffle=True, num_workers=opts.workers)
    val_dataloader = DataLoader(val_dataset, batch_size=opts.batch_size*2,
                            shuffle=False, num_workers=opts.workers)

    #checkpointing
    checkpoints_folder='checkpoints_fold{}'.format((opts.fold))
    csv_file='log_fold{}.csv'.format((opts.fold))
    columns=['epoch','train_loss',
             'val_loss','val_acc']
    logger=CSVLogger(columns,csv_file)

    #build model and logger
    model=NucleicTransformer(opts.ntoken, opts.nclass, opts.ninp, opts.nhead, opts.nhid,
                           opts.nlayers, opts.kmer_aggregation, kmers=opts.kmers,stride=opts.stride,
                           dropout=opts.dropout).to(device)
    optimizer=torch.optim.Adam(model.parameters(), weight_decay=opts.weight_decay)
    criterion=nn.CrossEntropyLoss(reduction='none')
    lr_schedule=lr_AIAYN(optimizer,opts.ninp,opts.warmup_steps,opts.lr_scale)
    softmax = nn.Softmax(dim=1)

    # Mixed precision initialization
    opt_level = 'O1'
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    model = nn.DataParallel(model)


    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total number of paramters: {}'.format(pytorch_total_params))

    #print("Starting training for fold {}/{}".format(opts.fold,opts.nfolds))
    #training loop
    for epoch in range(opts.epochs):
        model.train(True)
        t=time.time()
        total_loss=0
        optimizer.zero_grad()
        train_preds=[]
        ground_truths=[]
        step=0
        for data in dataloader:
        #for step in range(1):
            step+=1
            lr=lr_schedule.step()
            src=data['data'].long()
            labels=data['labels']
            #src=mutate_dna_sequence(src,opts.nmute)
            src=src.to(device).long()

            labels=labels.to(device).long()
            output=model(src,None)
            loss=torch.mean(criterion(output,labels))

            with amp.scale_loss(loss, optimizer) as scaled_loss:
               scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            total_loss+=loss
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.3f} Lr:{:.6f} Time: {:.1f}"
                           .format(epoch+1, opts.epochs, step+1, len(dataloader), total_loss/(step+1) , lr,time.time()-t),end='\r',flush=True) #total_loss/(step+1)
            #break
        print('')
        train_loss=total_loss/(step+1)
        #recon_acc=np.sum(recon_preds==true_seqs)/len(recon_preds)
        torch.cuda.empty_cache()
        if (epoch+1)%opts.val_freq==0:
            val_loss,val_acc=validate(model,device,val_dataloader,batch_size=opts.batch_size)
            to_log=[epoch+1,train_loss,val_loss,val_acc,]
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
