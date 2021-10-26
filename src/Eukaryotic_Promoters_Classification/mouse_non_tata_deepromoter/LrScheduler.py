import numpy as np


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class lr_AIAYN():
    '''
    Learning rate scheduler from the paper:
    Attention is All You Need
    '''
    def __init__(self,optimizer,d_model,warmup_steps=4000,factor=1):
        self.optimizer=optimizer
        self.d_model=d_model
        self.warmup_steps=warmup_steps
        self.step_num=0
        self.factor=factor

    def step(self):
        self.step_num+=1
        lr=self.d_model**-0.5*np.min([self.step_num**-0.5,
                                      self.step_num*self.warmup_steps**-1.5])*self.factor
        update_lr(self.optimizer,lr)
        return lr
        
        
class Cos_Anneal():
    '''
    Learning rate scheduler flat and anneal
    '''
    def __init__(self,optimizer,max_lr,min_lr,T):
        self.optimizer=optimizer
        self.max_lr=max_lr
        self.min_lr=min_lr
        self.step_num=0
        self.T=T

    def step(self):
        pi=3.1415
        self.step_num+=1
        lr=self.min_lr+0.5*(self.max_lr-self.min_lr)*(1+np.cos(self.step_num/self.T*pi))
        if self.optimizer:
            update_lr(self.optimizer,lr)
        return lr        