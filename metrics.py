import torch
from torch import nn
from sklearn.metrics import auc,roc_auc_score,log_loss
import numpy as np


def get_mlogloss(y_pred,y_true):
    mlogloss=[]
    dim=len(y_pred[0])
    for i in range(dim):
        tmp=0
        try:
            tmp=log_loss(y_true[:,i].astype('int'),y_pred[:,i].astype('float'))
        except:
            pass
        mlogloss.append(tmp)
    mlogloss=sum(mlogloss)/dim
    return mlogloss


def get_mlogloss2(y_pred,y_true,mask):

    dim=len(y_pred[0])

    new_y_pred=[]
    new_y_true=[]
    for i in range(len(y_pred)):
        if mask[i]>0:            #只计算有区域的logloss
            new_y_pred.append(y_pred[i])
            new_y_true.append(y_true[i])
    new_y_pred=np.array(new_y_pred)
    new_y_true=np.array(new_y_true)

    mlogloss=0
    for j in range(dim):
        tmp=0
        try:
            tmp=log_loss(new_y_true[:,j].astype('int'),new_y_pred[:,j].astype('float'))
        except:
            pass
        mlogloss+=tmp
    mlogloss=mlogloss/dim
    return mlogloss

class metrics_func(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = torch.nn.BCELoss()
        
    def forward(self, logits, labels):
        """
        logits:[batch_size,num_label]
        labels:[batch_size,num_label]
        """
        y_pred=logits.numpy()
        y_true=labels.numpy()

        mlogloss_region=get_mlogloss(y_pred[:,:17],y_true[:,:17])  #区域loss
        mask_tmp=torch.sum(labels,dim=1).view(-1,)
        mlogloss_type=get_mlogloss2(y_pred[:,17:-1],y_true[:,17:-1],mask_tmp) #类型loss
        
        return 1-2*mlogloss_region,1-2*mlogloss_type
        