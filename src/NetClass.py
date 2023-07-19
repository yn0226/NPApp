# -*- coding: utf-8 -*-

#必要なモジュール
#-----------------------------
#辞書インストール
#pip install unidic-lite
#↑がだめで、↓でインストールしてmecab使える
#python -m unidic download
#-----------------------------


import numpy as np
import pandas as pd

#import MeCab
#import oseti

from janome.tokenizer import Tokenizer
import re
import codecs

import csv
#------------------
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.functional import accuracy
#import torchsummary
#from torchsummary import summary
from pytorch_lightning.loggers import CSVLogger


# 前処理
# vectorizer = CountVectorizer(min_df=20)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ネットワークの定義
class Net(pl.LightningModule):

    # def __init__(self):
    #     super().__init__()
    #     self.fc1 = nn.Linear(133, 80) #辞書の数→中間層2の数
    #     self.fc2 = nn.Linear(100, 3)    #中間層の数→クラス数
        

    #     #(226, 133) (97, 133) BoW

        
    # def forward(self, x):
    #     h = self.fc1(x)
    #     h = F.relu(h)
    #     h = self.fc2(h) #中間層2まではここでおしまい        
    #     return h
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(133, 232) #辞書の数→中間層2の数
        self.relu1 =nn.ReLU()        
        self.fc2 = nn.Linear(232, 45)    #中間層2の数→中間層3の数
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(45, 3)    #中間層3の数→クラス数

        #(226, 133) (97, 133) BoW

        
    def forward(self, x):
        # h = self.fc1(x)
        # h = F.relu(h)
        # h = self.fc2(h) #中間層2まではここでおしまい
        # h = F.relu(h)   #中間層3を増やす
        # h = self.fc3(h)
        # return h
        h = self.relu1(self.fc1(x))
        h = self.relu2(self.fc2(h))
        h = self.fc3(h)
        return h