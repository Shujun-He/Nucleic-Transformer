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

def get_best_weights_from_fold(fold,top=3):
    csv_file='log_fold{}.csv'.format(fold)

    history=pd.read_csv(csv_file)
    scores=np.asarray(history.val_auc)
    top_epochs=scores.argsort()[-3:][::-1]
    print(scores[top_epochs])
    os.system('mkdir best_weights')

    for i in range(top):
        weights_path='checkpoints_fold{}/epoch{}.ckpt'.format(fold,history.epoch[top_epochs[i]])
        print(weights_path)
        os.system('cp {} best_weights/fold{}top{}.ckpt'.format(weights_path,fold,i+1))
    #os.system('rm -r checkpoints_fold{}'.format(fold))

def smoothcrossentropyloss(pred,gold,n_class=2,smoothing=0.05):
    gold = gold.contiguous().view(-1)
    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)
    loss = -(one_hot * log_prb)
    #loss=loss.sum(1).mean()
    return loss

def mutate_dna_sequence(sequence,nmute=15):
    to_mutate=np.random.choice(len(sequence[0]),size=(nmute),replace=False)
    mutation=np.random.randint(4,size=(len(sequence),nmute))
    mutated_sequence=sequence.copy()
    mutated_sequence[:,to_mutate]=mutation
    return mutated_sequence

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



def validate(model,device,dataset,batch_size=64):
    batches=int(len(dataset.val_indices)/batch_size)+1
    model.train(False)
    total=0
    ground_truths=dataset.labels[dataset.val_indices]
    predictions=[]
    outputs=[]
    loss=0
    criterion=nn.CrossEntropyLoss()
    dataset.switch_mode(training=False)
    dataset.update_batchsize(batch_size)
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            data=dataset[i]
            X=torch.Tensor(data['data']).to(device,).long()
            Y=torch.Tensor(data['labels']).to(device,dtype=torch.int64)
            #directions=data['directions']
            #directions=directions.reshape(len(directions),1)*np.ones(X.shape)
            #directions=torch.Tensor(directions).to(device).long()
            output,_,= model(X,None)
            del X
            loss+=criterion(output,Y)
            classification_predictions = torch.argmax(output,dim=1).squeeze()
            for pred in classification_predictions:
                predictions.append(pred.cpu().numpy())
            for vector in output:
                outputs.append(vector.cpu().numpy())
            del output
    torch.cuda.empty_cache()
    val_loss=(loss/batches).cpu()
    predictions=np.asarray(predictions)
    outputs=np.asarray(outputs)
    binary_predictions=predictions.copy()
    binary_predictions[binary_predictions==2]=1
    binary_ground_truths=ground_truths.copy()
    binary_ground_truths[binary_ground_truths==2]=1
    #print(predictions)
    #print(ground_truths)
    #score=metrics.cohen_kappa_score(ground_truths,predictions,weights='quadratic')
    val_acc=Metrics.accuracy(predictions,ground_truths)
    auc=metrics.roc_auc_score(ground_truths,outputs[:,1])
    val_sens=Metrics.sensitivity(predictions,ground_truths)
    val_spec=Metrics.specificity(predictions,ground_truths)
    binary_acc=np.sum(binary_predictions==binary_ground_truths)/len(binary_ground_truths)
    print('Accuracy: {}, Binary Accuracy: {} Val Loss: {}'.format(val_acc,binary_acc,val_loss))
    return val_loss,auc,val_acc,val_sens,val_spec


def predict(model,device,dataset,batch_size=64):
    batches=int(len(dataset.val_indices)/batch_size)+1
    model.train(False)
    total=0
    ground_truths=dataset.labels[dataset.val_indices]
    predictions=[]
    attention_weights=[]
    loss=0
    criterion=nn.CrossEntropyLoss()
    dataset.switch_mode(training=False)
    dataset.update_batchsize(batch_size)
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            data=dataset[i]
            X=torch.Tensor(data['data']).to(device,).long()
            Y=torch.Tensor(data['labels']).to(device,dtype=torch.int64)
            directions=data['directions']
            directions=directions.reshape(len(directions),1)*np.ones(X.shape)
            directions=torch.Tensor(directions).to(device).long()
            output,_,_,aw= model(X,directions,None)
            del X
            loss+=criterion(output,Y)
            classification_predictions = torch.argmax(output,dim=1).squeeze()
            for pred in output:
                predictions.append(pred.cpu().numpy())
            for weight in aw:
                attention_weights.append(weight.cpu().numpy())

            del output
    torch.cuda.empty_cache()
    val_loss=(loss/batches).cpu()
    predictions=np.asarray(predictions)
    attention_weights=np.asarray(attention_weights)
    binary_predictions=predictions.copy()
    binary_predictions[binary_predictions==2]=1
    binary_ground_truths=ground_truths.copy()
    binary_ground_truths[binary_ground_truths==2]=1
    return predictions,attention_weights,np.asarray(dataset.data[dataset.val_indices])
