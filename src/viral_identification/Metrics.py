import numpy as np


def accuracy(predictions,ground_truths):
    return np.sum(predictions==ground_truths)/len(ground_truths)
    
    
def sensitivity(predictions,ground_truths):
    '''
    Here it is assumed:
    0=negative
    1=positive
    '''
    return 1-len(predictions[(predictions==0)*(ground_truths==1)])/len(ground_truths[ground_truths==1])



def specificity(predictions,ground_truths):
    '''
    Here it is assumed:
    0=negative
    1=positive
    '''
    return 1-len(predictions[(predictions==1)*(ground_truths==0)])/len(ground_truths[ground_truths==0])
   
def MCC(predictions,ground_truths):
    '''
    Here it is assumed:
    0=negative
    1=positive
    '''
    N1=len(predictions[(predictions==0)&(ground_truths==1)])
    N2=len(predictions[(predictions==1)&(ground_truths==0)])
    N3=len(ground_truths[ground_truths==1])
    N4=len(ground_truths[ground_truths==0])
    sens=1-N1/N3
    spec=1-N2/N4
    denom=np.sqrt((1+(N2-N1)/N3)*(1+(N1-N2)/N4))
    return (1-sens-spec)/denom
    
    
    