#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, f1_score, roc_auc_score, recall_score
from sklearn.model_selection import train_test_split, cross_validate
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# In[2]:


def train_with_reg_cv( trX, trY, vaX, vaY, teX, teY, penalty='l1',
        C=2**np.arange(-8, 1).astype(np.float), seed=42):
    scores = []
    for i, c in enumerate(C):
        model = LogisticRegression(C=c,solver = 'liblinear',class_weight = 'balanced',penalty=penalty, random_state=seed+i)
#         model = linear_model.LinearRegression()
        model.fit(trX, trY)
        score = model.score(vaX, vaY)
        scores.append(score)
    c = C[np.argmax(scores)]
    model = LogisticRegression(C=c, penalty=penalty, class_weight = 'balanced',solver = 'liblinear' ,random_state=seed+len(C))
#     model = linear_model.LinearRegression()
    model.fit(trX, trY)
    
    if teX is not None and teY is not None:
        score = model.score(teX, teY)*100.
    else:
        score = model.score(vaX, vaY)*100.
    return score, c, model


# In[3]:


class ConvNet(nn.Module):
    def __init__(self, out_dim, fc_dim):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        self.fc = nn.Linear(fc_dim, out_dim)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


# In[5]:


class Conv_CCA(nn.Module):
    def __init__(self, out_dim):
        super(Conv_CCA, self).__init__()
        self.cnn1 = ConvNet(out_dim, fc_dim = 9741)
        self.cnn2 = ConvNet(out_dim, fc_dim = 4584)
        
    def forward(self, atmat, vtmat):
        output1 = self.cnn1(atmat)
        output2 = self.cnn2(vtmat)
        return output1, output2



class Conv_CCAA(nn.Module):
    def __init__(self, out_dim):
        super(Conv_CCAA, self).__init__()
        self.cnn1 = ConvNet(out_dim, fc_dim = 9741)
        self.cnn2 = ConvNet(out_dim, fc_dim = 6303)
        
    def forward(self, atmat, vtmat):
        output1 = self.cnn1(atmat)
        output2 = self.cnn2(vtmat)
        return output1, output2

def idx_finder(l):
    i = 0
    a = 0
    while a < (sum(l) * 0.95):
        a += l[i]
        i +=1
    return i

class Conv_SVCCA(nn.Module):
    def __init__(self, out_dim):
        super(Conv_SVCCA, self).__init__()
        self.cnn1 = ConvNet(out_dim, fc_dim = 9741)
        self.cnn2 = ConvNet(out_dim, fc_dim = 6303)
        
    def forward(self, atmat, vtmat):
        output1 = self.cnn1(atmat)
        output2 = self.cnn2(vtmat)
        o1 = output1 + torch.eye(output1.shape[0], output1.shape[1]) * 1e-4
        u1,s1,v1 = torch.svd(o1)
        idx1 = idx_finder(s1)
        o2 = output2 + torch.eye(output2.shape[0], output2.shape[2]) * 1e-4
        u2,s2,v2 = torch.svd(output2)
        idx2 = idx_finder(s2)
        svd1 = Variable(torch.matmul(u1[:,0:idx1], torch.diag(s[0:idx1])), requires_grad = True)
        svd2 = Variable(torch.matmul(u1[:,0:idx2], torch.diag(s[0:idx1])), requires_grad = True)
	
        return svd1, svd2


# In[4]:


def total(params):
    '''
    count the total number of hyperparameter settings
    '''
    settings = 1
    for k, v in params.items():
        settings *= len(v)
    return settings


class Regressor(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Regressor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
	    nn.Linear(hidden_dim, hidden_dim),
	    nn.ReLU(),
            nn.Linear(hidden_dim,1))
        
    def forward(self, in_vector):
        output = self.layer(in_vector)
        
        return output

class Regressor2(nn.Module):
    def __init__(self, in_dim_1, in_dim_2, hidden_dim, out_dim):
        super(Regressor2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim_1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,out_dim))
        self.layer2 = nn.Sequential(
            nn.Linear(in_dim_2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,out_dim))
    def forward(self, in1, in2):
        out1 = self.layer1(in1)
        out2 = self.layer2(in2)
        return out1, out2

import torch.nn.functional as F
class Binarycls(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Binarycls, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
	    nn.Linear(hidden_dim, hidden_dim),
	    nn.ReLU(),
            nn.Linear(hidden_dim,2))
        
    def forward(self, in_vector):
        output = F.sigmoid(self.layer(in_vector))
        
        return F.log_softmax(output)


