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

# import sys
# from sklearn import feature_extraction
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import CountVectorizer

import metrics

def read_data(path,mode):
    sort_labels = [19, 15,  4,  0, 11,  7, 17, 22,  1,  8,  9,  2, 16, 14, 28, 26, 27, 24,  3, 10,  5, 12, 21, 13,  6, 25, 20, 18, 23]
    label_2_ids = dict(zip(range(29),sort_labels))
    ids_2_label = dict(zip(sort_labels,range(29)))
    
    data = []
    file = open(path,'r')
    for x in file.readlines():
        data.append(x.strip().split('|,|'))
    file.close()
    data = pd.DataFrame(data)
    data[0] = data[0].map(int)
    data[1] = data[1].apply(lambda x:x[:-1]).str.split(' ')
    data[1] = data[1].apply(lambda x: [a for a in x if a != ''])
    data = data.rename(columns={0:'report_ID',1:'description'})
    if mode == 'train':
        data = data.join(data[2].str.split(',',expand=True))
        data[0] = data[0].str.split(' ')
        data[1] = data[1].str.split(' ')
        data[0] = data[0].apply(lambda x:[] if x==[''] else [int(temp) for temp in x if temp != ''])
        data[1] = data[1].apply(lambda x:[] if x==[''] else [int(temp) for temp in x if temp != ''])
        data[2] = data[0] + data[1].apply(lambda x: [temp+17 for temp in x])
        data[2] = data[2].apply(lambda x: sorted([label_2_ids[temp] for temp in x]))
        data[2] = data[2].apply(lambda x: [ids_2_label[temp] for temp in x])
        data = data.rename(columns={0:'label1',1:'label2',2:'label'})
    return data

def read_d(path):
    data = []
    file = open(path,'r')
    for x in file.readlines():
        data.append(x.strip().split('|,|'))
    file.close()
    data = pd.DataFrame(data)
    data[0] = data[0].map(int)
    data[1] = data[1].apply(lambda x:x[:-1]).str.split(' ')
    data[1] = data[1].apply(lambda x: [a for a in x if a != ''])
    data = data.rename(columns={0:'report_ID',1:'description'})
    return data

def TF_IDF(data,tfidf_model,dictionary,id2token,token2id):
    temp = tfidf_model[dictionary.doc2bow(data)]
    temp = dict(temp)
    out = []
    for x in data:
        out.append(temp[token2id[x]])
#     x = torch.LongTensor([id2token[i[0]] for i in np.array(temp)[:,:1].tolist()]).view(-1,1)#.to(DEVICE)
#     y = torch.FloatTensor(temp)[:,1:].view(1,-1)#.to(DEVICE)
#     x = torch.zeros(len(x), 859).scatter_(1, x,torch.ones(len(x), 859))
#     out = torch.matmul(y.float(),x.float()).view(-1)
    return out

def data_json(data, mode):
    output = []
    if mode == 'train':
        for i,row in data.iterrows():
            output_li = {'text':row['description'], 'label':row['label'],'TF_IDF':row['TF_IDF']}
            output.append(output_li)
    else:
        for i,row in data.iterrows():
            output_li = {'text':row['description'], 'label':[],'TF_IDF':row['TF_IDF']}
            output.append(output_li)
    return output

class My_data(object):
    def __init__(self):
        super().__init__()
    def forward(train_data_path,data1_path,data2_path,data3_path,test_data_path,embedding_path):
        train_data = read_data(train_data_path,'train')
        test_data = read_data(test_data_path,'test')
        data1 = read_d(data1_path)
        data2 = read_d(data2_path)
        data3 = read_d(data3_path)
        
        all_sentences = pd.concat([train_data['description'], data1['description'], data2['description'], data3['description']]).reset_index(drop=True)
        corpora_documents = all_sentences.tolist()
        dictionary = corpora.Dictionary(corpora_documents)
        corpus = [dictionary.doc2bow(text) for text in corpora_documents]
        tfidf_model=models.TfidfModel(corpus)
        id2token = {y:int(x)+1 for x,y in  dictionary.token2id.items()}
        token2id = {x:y for x,y in  dictionary.token2id.items()}
        
#         vectorizer=CountVectorizer()
#         transformer=TfidfTransformer()
#         tfidf=transformer.fit_transform(vectorizer.fit_transform(all_sentences.tolist()))
#         word=vectorizer.get_feature_names()
#         weight=tfidf.toarray()
#         print(all_sentences[0])
#         print(word[:50])
#         print(weight[0][:50])
        train_data['TF_IDF'] = train_data['description'].map(lambda x : TF_IDF(x,tfidf_model,dictionary,id2token,token2id))
        test_data['TF_IDF'] = test_data['description'].map(lambda x : TF_IDF(x,tfidf_model,dictionary,id2token,token2id))
        
        train_json = data_json(train_data,'train')
        test_json = data_json(test_data,'test')
        
        if not os.path.exists(embedding_path+'w2v.model'): 
            w2v = gensim.models.word2vec.Word2Vec(all_sentences, sg=1, vector_size=300, window=5, min_count=1, negative=3,
                                                        seed=666, sample=0.001, hs=1)
            w2v.save(embedding_path+'w2v.model')
        else:
            w2v = gensim.models.word2vec.Word2Vec.load(embedding_path+"w2v.model")

        if not os.path.exists(embedding_path+'fasttext.model'): 
            fasttext_model = gensim.models.FastText(all_sentences, seed=666, vector_size=100, min_count=1, epochs=20, window=2)
            fasttext_model.save(embedding_path+'fasttext.model')
        else:
            fasttext_model = gensim.models.word2vec.Word2Vec.load(embedding_path+"fasttext.model")
            
        return train_data,test_data,train_json,test_json,w2v,fasttext_model
        