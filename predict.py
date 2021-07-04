#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn 
import random
import gensim
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.utils import data
import tqdm.auto as tqdm
from torch.optim import *
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, classification_report
from transformers import  BertModel
from sklearn.model_selection import KFold
import collections
import os
import random
from gensim.models import word2vec
import time
# import sys
# from sklearn import feature_extraction
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import log_loss
import metrics

EPOCHS = 13
EARLY_STOP = 10
FOLD = 10
SEED = 9797
MAX_LEN = 115
LABEL_NUM = 29
MAX_LEN_LABEL = LABEL_NUM+1
BATCH_SIZE = 16
HIDDEN_SIZE = 768
WORD_EMBEDDING = 550

embedding_path = '../../user_data/'
START = time.time()

model_path = 'model'
result_path = 'result'


class Config(object):
    """配置参数"""
    def __init__(self):
        self.model_name = 'seq2seq'
#         self.embeddings = True                                         
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        self.dropout = 0.1
        self.input_dim = 859
        self.emb_dim = WORD_EMBEDDING                                  
        self.hid_dim = HIDDEN_SIZE
        self.n_layers = 1
        self.output_dim = LABEL_NUM
        self.part1 = 17
        self.part2 = 12
        self.vocab = 859
        self.label_embedding = 10


# In[2]:


def seed_everything(SEED):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if DEVICE=='cuda':
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return DEVICE
DEVICE = seed_everything(SEED)


# In[3]:


import logging

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# In[4]:


import Mydata
train_path = '../../tcdata/train.csv'
test_path = '../../tcdata/testB.csv'
data1_path = '../../tcdata/track1_round1_train_20210222.csv'
data2_path = '../../tcdata/track1_round1_testA_20210222.csv'
data3_path = '../../tcdata/track1_round1_testB.csv'
do_data = Mydata.My_data
train_data,test_data,train_json,test_json,w2v,fasttext_model = do_data.forward(train_path,data1_path,data2_path,data3_path,test_path,embedding_path)


# In[5]:



class DataSet(data.Dataset):
    def __init__(self, data, mode='train'):
        self.data = data
        self.mode = mode
        self.dataset = self.get_data(self.data, self.mode)
        
    def get_data(self, data, mode):
        dataset = []
        for data_li in tqdm.tqdm(data):   #data_li是dict
            TF = data_li['TF_IDF'][:MAX_LEN]
            text = data_li['text'][:MAX_LEN]
#             temp_dict = w2v.wv.key_to_index
#             sen = [temp_dict[s]+1 for s in text]
            sen = [int(x)+1 for x in text]
            label  = [LABEL_NUM+2]+data_li['label']
            label_01 = [0]*LABEL_NUM
            for x in data_li['label']:
                label_01[x] = 1
            if len(sen) < MAX_LEN:
                sen = sen + [0] * (MAX_LEN - len(sen))
                TF = TF + [0] * (MAX_LEN - len(TF))
            if len(label) < MAX_LEN_LABEL:
                label = label + [LABEL_NUM+1] * (MAX_LEN_LABEL - len(label))
            dataset_dict = {'sen':sen,
                            'TF_IDF':TF,
                            'label':label,
                            'label_01':label_01
                           }
            dataset.append(dataset_dict)
        return dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        sen = torch.tensor(data['sen'])
        label = torch.tensor(data['label'])
        label_01 = torch.tensor(data['label_01'])
        TF_IDF = torch.tensor(data['TF_IDF']).float()
        return sen, label, label_01, TF_IDF
    
def get_dataloader(dataset, mode):
    torchdata = DataSet(dataset, mode=mode)
    if mode == 'train':
        dataloader = torch.utils.data.DataLoader(torchdata, batch_size=BATCH_SIZE, 
                                                 shuffle=True, num_workers=0, drop_last=True)
    elif mode == 'test':
        dataloader = torch.utils.data.DataLoader(torchdata, batch_size=BATCH_SIZE, 
                                                 shuffle=False, num_workers=0, drop_last=False)
    elif mode == 'valid':
        dataloader = torch.utils.data.DataLoader(torchdata, batch_size=BATCH_SIZE, 
                                                 shuffle=False, num_workers=0, drop_last=False)
    return dataloader, torchdata
# data21, data22 = get_dataloader(test_json[:300], mode='test')
# data22.dataset[0]


# In[6]:


def validation_funtion(model, valid_dataloader, valid_torchdata, mode):
    model.eval()
    prediction = []
    true_results = []
    for i, (sen, labels,label_01,tf_idf) in enumerate(tqdm.tqdm(valid_dataloader)):
        predic = model(sen.to(DEVICE),labels.long().to(DEVICE), label_01.to(DEVICE),tf_idf.to(DEVICE), mode=mode, teacher_forcing_ratio=0).cpu().detach().numpy().tolist()
        label_ids = label_01.cpu().detach().numpy().tolist()
#         print(len(predic),len(label_ids))
        prediction += list(predic)
        true_results += list(label_ids)
    if mode == 'test':
        return prediction, true_results
    else:
        true_results = np.array(true_results)#[100,17]
        prediction = np.array(prediction)
        duiyou = metrics.metrics_func()
        score1,score2 = duiyou(torch.tensor(prediction),torch.tensor(true_results))
        return score1,score2    
                            
def train(model, train_dataloader, valid_dataloader, valid_torchdata, epochs, early_stop):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.08},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-4, amsgrad=True, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2, eta_min=1e-4, last_epoch=-1)
#     scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=3,T_mult=2,eta_min=1e-5)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma = 0.1)
#     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=150, num_training_steps=num_training_steps)
    total_loss = []
    train_loss = []
    best_f1 = -np.inf
    no_improve = 0
    for epoch in range(epochs):
        model.train()
        bar = tqdm.tqdm(train_dataloader)
        for i, (sen, labels, label_01,tf_idf) in enumerate(bar):
            loss = model(sen.to(DEVICE),labels.long().to(DEVICE),label_01.long().to(DEVICE),tf_idf.to(DEVICE), mode='train')
            loss.backward()
            train_loss.append(loss.item())
            optimizer.step()
            scheduler.step(epoch + i / len(train_dataloader))
            
            optimizer.zero_grad()
            bar.set_postfix(tloss=np.array(train_loss).mean())
        score1,score2 = validation_funtion(model, valid_dataloader, valid_torchdata, 'valid')
        score = 0.6*score1 + 0.4*score2
        logger.info('EPOCHE:[{}], train_loss: {}, score1: {}, score2: {}, score: {}\n'.format(epoch, np.mean(train_loss),score1,score2,score))
        global model_num
        if early_stop:
            if score > best_f1:
                best_f1 = score
                torch.save(model.state_dict(), 'model_{}.bin'.format(model_num))
            else:
                no_improve += 1
            if no_improve == early_stop:
                model_num += 1
                logger.info('EPOCHE:[{}], best_f1: {}\n'.format(epoch, best_f1))
                break
            if epoch == epochs-1:
                model_num += 1
        else:
            if epoch >= epochs-1:
                torch.save(model.state_dict(), 'model_{}.bin'.format(model_num))
                model_num += 1
                
#         if score > best_f1:
#             best_f1 = score
#             torch.save(model.state_dict(), 'model_{}.bin'.format(model_num))
#     logger.info('EPOCHE:[{}], best_f1: {}\n'.format(epoch, best_f1))
#     model_num += 1


# ## predict

# In[7]:


import Mynet
model_num = 11
config = Config()
model = Mynet.My_net.forward(config,w2v,fasttext_model,DEVICE)
model.to(DEVICE)
test_preds_total = []
test_dataloader, test_torchdata = get_dataloader(test_json, mode='test')
for i in range(1,model_num):
    model.load_state_dict(torch.load('model_{}.bin'.format(i)))
    test_pred_results, _ = validation_funtion(model, test_dataloader, test_torchdata, 'test')
    test_preds_total.append(test_pred_results)


# In[8]:


test_preds_merge = pd.DataFrame(test_preds_total[0])/(model_num-1)
for i in range(1,model_num-1):
    df = pd.DataFrame(test_preds_total[i])
    test_preds_merge += df/(model_num-1)
test_preds_merge


# In[9]:


submit = test_data[['report_ID']].copy()
submit['all_data'] = test_preds_merge[0].map(str)
for x in range(1,LABEL_NUM):
    submit['all_data'] += ' '
    submit['all_data'] += test_preds_merge[x].map(str)
submit


# In[10]:


str_w=''
sub_id=test_data['report_ID'].values
with open('../../result8.csv','w') as f:
    for i in range(len(submit['report_ID'])):
        str_w+=f'{i}'+'|,'+'|'+submit['all_data'][i]+'\n'
    str_w=str_w.strip('\n')
    f.write(str_w)

