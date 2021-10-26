import torch
import os
from sklearn import metrics
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import Metrics
import numpy as np
import os
import pandas as pd
import torch
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef,precision_score,recall_score

#f1_score(y_true, y_pred

def iter_split_strict(data,labels,fold,nfolds=5,seed=2020):
    splits = StratifiedKFold(n_splits=nfolds, random_state=seed, shuffle=True)
    splits = list(splits.split(data,labels))
    # splits = np.zeros(len(data)).astype(np.int)
    # for i in range(nfolds): splits[splits[i][1]] = i
    # indices=np.arange(len(data))
    train_indices=splits[fold][0]
    train_indices, val_indices=train_test_split(train_indices,test_size=0.25)
    test_indices=splits[fold][1]
    return train_indices, val_indices, test_indices


def iter_split(data,labels,fold,nfolds=5,seed=2020):
    splits = StratifiedKFold(n_splits=nfolds, random_state=seed, shuffle=True)
    splits = list(splits.split(data,labels))
    # splits = np.zeros(len(data)).astype(np.int)
    # for i in range(nfolds): splits[splits[i][1]] = i
    # indices=np.arange(len(data))
    train_indices=splits[fold][0]
    val_indices=splits[fold][1]
    return train_indices, val_indices

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(seed=42)

def get_best_weights_from_fold(fold,top=1):
    csv_file='logs/log_fold{}.csv'.format(fold)

    history=pd.read_csv(csv_file)
    scores=np.asarray(history.val_acc)
    top_epochs=scores.argsort()[-3:][::-1]
    print(scores[top_epochs])
    os.system('mkdir best_weights')

    for i in range(top):
        weights_path='checkpoints_fold{}/epoch{}.ckpt'.format(fold,history.epoch[top_epochs[i]])
        print(weights_path)
        os.system('cp {} best_weights/fold{}top{}.ckpt'.format(weights_path,fold,i+1))
    os.system('rm -r checkpoints_fold{}'.format(fold))

def smoothcrossentropyloss(pred,gold,n_class=2,smoothing=0.05):
    gold = gold.contiguous().view(-1)
    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)
    loss = -(one_hot * log_prb)
    #loss=loss.sum(1).mean()
    return loss

def mutate_dna_sequence(sequence,nmute=15):
    mutation=torch.randint(0,4,size=(sequence.shape[0],nmute))
    to_mutate = torch.randperm(sequence.shape[1])[:nmute]
    sequence[:,to_mutate]=mutation
    return sequence

def get_MLM_mask(sequence,nmask=12):
    mask=np.zeros(sequence.shape,dtype='bool')
    to_mask=np.random.choice(len(sequence[0]),size=(nmask),replace=False)
    mask[:,to_mask]=True
    return mask

def get_complementary_sequence(sequence):
    complementary_sequence=sequence.copy()
    complementary_sequence[sequence==0]=1
    complementary_sequence[sequence==1]=0
    complementary_sequence[sequence==2]=3
    complementary_sequence[sequence==3]=2
    complementary_sequence=complementary_sequence[:,::-1]
    return complementary_sequence

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_weights(model,optimizer,epoch,folder):
    if os.path.isdir(folder)==False:
        os.makedirs(folder,exist_ok=True)
    torch.save(model.state_dict(), folder+'/epoch{}.ckpt'.format(epoch+1))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        lr=param_group['lr']
    return lr

def validate(model,device,dataset,batch_size=64):
    batches=len(dataset)
    model.train(False)
    total=0
    ground_truths=[]
    predictions=[]
    loss=0
    criterion=nn.CrossEntropyLoss()
    # dataset.switch_mode(training=False)
    # dataset.update_batchsize(batch_size)
    with torch.no_grad():
        for data in tqdm(dataset):
            #data=dataset[i]
            X=data['data'].to(device).float()
            #X=torch.nn.functional.one_hot(X,num_classes=4)
            Y=data['labels'].to(device).long()
            output= model(X)
            del X
            loss+=criterion(output,Y)
            classification_predictions = torch.argmax(output,dim=1).squeeze()
            for pred in classification_predictions:
                predictions.append(pred.cpu().numpy())
            for truth in Y:
                ground_truths.append(truth.cpu().numpy())
            del output
    ground_truths=np.asarray(ground_truths)
    torch.cuda.empty_cache()
    val_loss=(loss/batches).cpu()
    predictions=np.asarray(predictions)
    binary_predictions=predictions.copy()
    binary_predictions[binary_predictions==2]=1
    binary_ground_truths=ground_truths.copy()
    binary_ground_truths[binary_ground_truths==2]=1
    #print(predictions)
    #print(ground_truths)
    #score=metrics.cohen_kappa_score(ground_truths,predictions,weights='quadratic')
    val_acc=Metrics.accuracy(predictions,ground_truths)
    val_sens=Metrics.sensitivity(predictions,ground_truths)
    val_spec=Metrics.specificity(predictions,ground_truths)
    val_precision=precision_score(predictions,ground_truths)
    val_recall=recall_score(predictions,ground_truths)
    binary_acc=np.sum(binary_predictions==binary_ground_truths)/len(binary_ground_truths)
    val_f1=f1_score(ground_truths, predictions)
    val_mcc=matthews_corrcoef(ground_truths, predictions)
    print('Accuracy: {}, Binary Accuracy: {} Val F1: {} Val Loss: {}'.format(val_acc,binary_acc,val_f1,val_loss))
    return val_loss,val_acc,val_precision,val_recall,val_f1,val_mcc


def predict(model,device,dataset,batch_size=64):
    batches=len(dataset)
    model.train(False)
    total=0
    ground_truths=[]
    predictions=[]
    attention_weights=[]
    sequences=[]
    loss=0
    criterion=nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in tqdm(dataset):
            #data=dataset[i]
            X=data['data'].to(device,).float()
            Y=data['labels'].to(device,dtype=torch.int64)

            output= model(X)
            #del X
            loss+=criterion(output,Y)
            classification_predictions = torch.argmax(output,dim=1).squeeze()
            for pred in output:
                predictions.append(pred.cpu().numpy())
            # for weight in aw:
            #     attention_weights.append(weight.cpu().numpy())

            for t in Y:
                ground_truths.append(t.cpu().numpy())
            for seq in X:
                sequences.append(seq.cpu().numpy())
            del output
    torch.cuda.empty_cache()
    val_loss=(loss/batches).cpu()
    predictions=np.asarray(predictions)
    attention_weights=np.asarray(attention_weights)
    binary_predictions=predictions.copy()
    binary_predictions[binary_predictions==2]=1
    binary_ground_truths=ground_truths.copy()
    binary_ground_truths[binary_ground_truths==2]=1
    return predictions,attention_weights,np.asarray(sequences),np.asarray(ground_truths)
