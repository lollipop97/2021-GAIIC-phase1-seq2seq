import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn 
import random
import gensim
import copy
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils import data
import tqdm.auto as tqdm
from torch.optim import *
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
from sklearn.metrics import roc_auc_score, classification_report
from transformers import BertModel
from sklearn.model_selection import KFold
import collections
from gensim.models import word2vec
from sklearn.metrics import log_loss
from gensim import corpora, models
from gensim.similarities import Similarity
# seq2seq
import torch.nn.functional as F
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, config,DEVICE):
        super().__init__()
        self.device = DEVICE
        self.encoder = encoder
        self.decoder = decoder
        self.output_dim = config.output_dim
        
    def forward(self, src, trg, trg_01, tf_idf, mode = 'train',teacher_forcing_ratio=0.6):
        batch_size = src.shape[0]
        max_len = self.output_dim+1
        outputs = torch.zeros(batch_size,self.output_dim).to(self.device)
        hidden = self.encoder(src, tf_idf)
        Input = trg[:,0]
        for t in range(1, max_len):
            output, hidden = self.decoder.forward(Input, hidden)
            outputs += output/self.output_dim
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            Input = trg[:,t] if teacher_force else top1
        if mode == 'train':
            loss_fn = nn.BCEWithLogitsLoss()
            loss1 = loss_fn(outputs[:,:17].contiguous().view(-1).float(),trg_01[:,:17].contiguous().view(-1).float())
            loss2 = loss_fn(outputs[:,17:].contiguous().view(-1).float(),trg_01[:,17:].contiguous().view(-1).float())
            loss = loss1*1.1+loss2
            return loss
        else:
            sig = nn.Sigmoid() #将输出变为0-1之间
            pred = sig(outputs.detach().cpu())
            return pred


# Encoder
class Encoder(nn.Module):
    def __init__(self, config,w2v,fasttext,DEVICE):
        super().__init__()
        self.device = DEVICE
        self.input_dim = config.input_dim
        self.hid_dim = config.hid_dim
        self.n_layers = config.n_layers
        self.vocab = config.vocab
        self.embedding = nn.Embedding(config.vocab, config.emb_dim)
#         w2v_model = w2v.wv
#         fasttext_model = fasttext.wv
#         w2v_embed_matrix = w2v_model.vectors
#         fasttext_embed_matrix = fasttext_model.vectors
#         embed_matrix = np.concatenate([w2v_embed_matrix, fasttext_embed_matrix], axis=1)
#         oov_embed = np.zeros((1, embed_matrix.shape[1]))
#         embed_matrix = torch.from_numpy(np.vstack((oov_embed,embed_matrix)))
#         self.embedding.weight.data.copy_(embed_matrix)
#         self.embedding.weight.requires_grad = False
        
        self.rnn1 = nn.GRU(config.emb_dim,self.hid_dim, num_layers =self.n_layers, dropout=config.dropout, batch_first=True,bidirectional=True)
        self.rnn2 = nn.GRU(self.hid_dim*4+1, self.hid_dim,num_layers=self.n_layers, dropout=config.dropout,batch_first=True,bidirectional=True)
        self.dropout = nn.Dropout(config.dropout)
        
    def self_attention(self, x , mask=None):
        attention = torch.matmul(x, x.transpose(1, 2))
        weight = F.softmax(attention, dim=-1)
        x_align = torch.matmul(weight, x)
        return x_align
    
    def forward(self, src, tf_idf):
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)
        BS = src.shape[0]
        ML = src.shape[1]
        (outputs, _) = self.rnn1(embedded)
        outputs = self.dropout(outputs)
#         One_hot = torch.zeros(BS*ML ,self.vocab).to(self.device).scatter_(1, src.view(-1,1),torch.ones(BS*ML, self.vocab).to(self.device))
#         One_hot = One_hot.view(BS,ML,-1)
#         TF = torch.matmul(One_hot, tf_idf.unsqueeze(2).to(self.device))
        atten = self.self_attention(outputs)
        outputs_atten = torch.cat([outputs,atten,tf_idf.unsqueeze(2)], -1)
        (_, hidden) = self.rnn2(outputs_atten)
        hidden = self.dropout(hidden)
        return hidden

#decoder
class Decoder(nn.Module):
    def __init__(self, config,DEVICE):
        super().__init__()
        self.device = DEVICE
        self.hid_dim = config.hid_dim
        self.n_layers = config.n_layers
        self.embedding = nn.Embedding(config.output_dim+3, config.label_embedding)
        self.rnn = nn.GRU(config.label_embedding, self.hid_dim, self.n_layers,batch_first=True,bidirectional=True)
        self.out1 = nn.Linear(self.hid_dim*2, config.part1)
        self.out2 = nn.Linear(self.hid_dim*2, config.part2)
        self.dropout = nn.Dropout(config.dropout)
        

        self.softmax = nn.Softmax(dim=1)
    def forward(self, Input, hidden):
        embedded = Input.unsqueeze(1).long()
        embedded = self.dropout(self.embedding(embedded))
        (output,_ )= self.rnn(embedded,hidden)
#         output = self.dropout(output)
        prediction1 = self.out1(output.squeeze(1))
        prediction2 = self.out2(output.squeeze(1))
        prediction = torch.cat([prediction1,prediction2],-1)
        prediction = self.dropout(prediction)
        return prediction, hidden

class My_net(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(config,w2v,fasttext,DEVICE='cuda'):
        encoder = Encoder(config,w2v,fasttext,DEVICE)
        decoder = Decoder(config,DEVICE)
        model = Seq2Seq(encoder.to(DEVICE),decoder.to(DEVICE),config, DEVICE)
        return model