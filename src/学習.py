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

import MeCab
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
import torchsummary
from torchsummary import summary
from pytorch_lightning.loggers import CSVLogger

#------------------
# CSV から読み込み
#------------------
filename = 'C:/01_Amazon/AmazonApp/src/Result_価格コム.csv'
df_CSV =pd.read_csv(filename, encoding='shift-jis')

# CSVからレビュー部分を順に取り出し、reviewsに格納
reviews = []    # review内容
labels =[]      # 判定結果
for review in df_CSV['comment']:    
    reviews.append(review)
    #print('全レビュー数：', len(reviews))
for label in df_CSV['Result']:
    labels.append(label) 
    #print('全ラベル数：', len(labels))

# reviewsからレビューを順に取り出し、wordsに格納
#review = 'ドラム式洗濯乾燥機を、これまで２台(ナショナル(NA-VR1000)、日立(BD-S8800L))使用してきたことを含めてのレビューです。【デザイン】エッジの効いた箱型でいい。【使いやすさ】カラータッチパネルで使いやすい。説明書もいらない。洗剤や柔軟剤の残量不足が分かるなどスマホ連携も便利。【洗浄力】今までの他社製品と比較して、これが1番すごいと思いました。白色ワイシャツを日々洗濯していますが、普通に洗濯しただげでも、汚れ落ちが良く、白さが際立っています。抗菌ウルトラファインバブル洗浄EXの効果なのかは分かりませんが、洗濯するたび、すごいと思っています。【静音性】洗濯時は、静かです。乾燥時、それなりの音はしますが、他社も同じです。【サイズ】他社と同じサイズです。【機能・メニュー】当初、液体洗剤・柔軟剤の自動投入はいらないと思っていましたが、非常に便利。また、スマホ連携で外出先からもスタート出来て、帰宅時間に合わせられるのでいいと思います。乾燥フィルターのお手入れも楽です。【価格】北関東に居住していますが、2023年5月上旬、23万円(5年保証、設置、リサイクル、収集運搬費など全てコミコミ)で購入しました。【総評】購入後、約2週間を過ぎての総評です。抗菌ウルトラファインバブル洗浄EX、液体洗剤・柔軟剤自動投入、大型カラータッチパネルの東芝ドラム式洗濯乾燥機の最上位モデルを謳っているだけのことはあります。さすがです。他社の最上位モデルと比較し、価格面も考慮するなら、現時点でベストバイだと思います。'
corpus = []
i = 0
for review in reviews:  

    # 形態素解析により名詞、形容詞、動詞を抽出
    token = Tokenizer().tokenize(review)
    words = []

    for review in token:
        tkn = re.split('\t|,', str(review))
        # 名詞、形容詞、動詞で判定
        if tkn[1] in ['名詞','形容詞','動詞'] :
            words.append(tkn[0])      

    #print(words[:3])
    #print(len(words))
    corpus.append(' '.join(words))


print(len(corpus))
#print(len(corpus[1]))
#print(len(corpus[2]))
    
#-----------------------------

#------------------
# データの分割
#------------------
#trainとtestデータに分割する
text_train_val, text_test,  t_train_val, t_test = train_test_split(corpus, labels, test_size=0.3, random_state=0, stratify=np.array(labels))
#↑stratifyの引数で目標値のクラスが均等になるように分割する
print(len(text_train_val), len(text_test))



#------------------
# 特徴量に変換
#------------------
vectorizer = CountVectorizer(min_df=20)
bow_train_val = vectorizer.fit_transform(text_train_val).toarray()
bow_test = vectorizer.transform(text_test).toarray()

print(bow_train_val.shape, bow_test.shape)

t_train_val = np.array(t_train_val)
t_test = np.array(t_test)
features = vectorizer.get_feature_names_out()
print('次元数:', len(features))

#------------------
# PyTorch で扱えるデータ形式に変換
#------------------

# tensor形式へ変換
x_train_val = torch.tensor(bow_train_val, dtype=torch.float32)
x_test = torch.tensor(bow_test, dtype=torch.float32)

t_train_val = torch.tensor(t_train_val, dtype=torch.int64)
t_test = torch.tensor(t_test, dtype=torch.int64)

# データセットにまとめる
dataset_train_val = torch.utils.data.TensorDataset(x_train_val, t_train_val)
dataset_test = torch.utils.data.TensorDataset(x_test, t_test)
print(len(dataset_train_val), len(dataset_test))     

# 今回の場合、テストデータはすでに分割しているので、学習用データと検証用データにだけ分割を行います。
pl.seed_everything(0)

# train と val に分割
n_train = int(len(dataset_train_val)*0.7)
n_val = int(len(dataset_train_val) - n_train)

train, val = torch.utils.data.random_split(dataset_train_val, [n_train, n_val])
print(len(train), len(val))

#------------------
# DataLoader の作成
#------------------
# バッチサイズの定義
batch_size = 34

# Data Loader を定義
train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val, batch_size)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size)

#------------------
# ネットワークの定義と学習
#------------------
class Net(pl.LightningModule):

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
    
    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy(y.softmax(dim=-1), t, task="multiclass",num_classes=3), on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', accuracy(y.softmax(dim=-1), t, task="multiclass",num_classes=3), on_step=False, on_epoch=True)
        return loss


    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', accuracy(y.softmax(dim=-1), t, task="multiclass",num_classes=3), on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=0.01) #SGD/ADAM/RAdam
        return optimizer
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = Net()
summary(net.to(device), input_size=(1, 133))    #133=辞書の数

#------------------
# 学習の実行
#------------------
pl.seed_everything(0)

net = Net()
logger = CSVLogger(save_dir='logs', name='my_exp')

trainer = pl.Trainer(max_epochs=20, accelerator="cpu", deterministic=False, logger=logger)
trainer.fit(net, train_loader, val_loader)

#------------------
# テストデータで検証
#------------------
results = trainer.test(dataloaders=test_loader)

#------------------
# モデルの保存
#------------------
# 学習済みモデルの保存
torch.save(net.state_dict(), './src/NPmodel.pt')

class Net(pl.LightningModule):

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
    
net = Net()
net.load_state_dict(torch.load('./src/NPmodel.pt'))
