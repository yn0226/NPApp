# -*- coding: utf-8 -*-

#必要なモジュール
#-----------------------------
#辞書インストール
#pip install unidic-lite
#↑がだめで、↓でインストールしてmecab使える
#python -m unidic download
#-----------------------------
#import sys
#sys.path.append('C:/Users/2nb23/anaconda3/Lib/site-packages')
#path_list = sys.path
#print(path_list)

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

#-------------------
# pythonファイル参照
#-------------------
from kakakuCom import Scrape, scrape_kakaku    # kakakuCom.py からスクレイピング定義を読み込み
from NetClass import Net # NetClass.py から前処理とネットワークの定義を読み込み

from flask import Flask, request, render_template, redirect
import io
import base64
from wtforms import Form, StringField, validators, SubmitField

# #------------------
# # スクレイピング
# #------------------
# # kakakuCom.pyから関数の呼び出し

# # スクレイピング処理
# res_URL = scrape_kakaku('https://review.kakaku.com/review/M0000000975/#tab')


# # データフレームの表示
# # kakakuCom.py から読み込み
# res_df = res_URL
# #print(res_df[:3])

# #------------------
# # 推論用レビュー
# #------------------
# #filename = '../価格com口コミ.csv'
# #df_Input =pd.read_csv(filename, encoding='shift-jis')
# df_Input = res_df

# # CSVからレビュー部分を順に取り出し、reviewsに格納
# reviews = []    # review内容
# for review in df_Input['comment']:    
#     reviews.append(review)
#     #print('全レビュー数：', len(reviews))

# corpus_In = []
# for review in reviews:  

#     # 形態素解析により名詞、形容詞、動詞を抽出
#     token = Tokenizer().tokenize(review)
#     words = []

#     for review in token:
#         tkn = re.split('\t|,', str(review))
#         # 名詞、形容詞、動詞で判定
#         if tkn[1] in ['名詞','形容詞','動詞'] :
#             words.append(tkn[0])      

#     #print(words[:3])
#     #print(len(words))
#     corpus_In.append(' '.join(words))

# print(len(corpus_In))

# #Vectorizer
# vectorizer = CountVectorizer(min_df=1, max_df=1.0)
# vectorizer.fit(corpus_In)
# pred_bow = vectorizer.transform(corpus_In).toarray()
# #Tensor
# pred_tensor = torch.tensor(pred_bow, dtype=torch.float32)
# pred_tensor = pred_tensor[:, :133]  # 学習済みモデルの133次元にスライスする

#------------------
# 推論
#------------------
# 学習済みモデルをもとに推論を行う
def predict(pred_tensor):
    print('推論:def predict1')
    #device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('推論:def predict2')

    # ネットワークの準備
    net = Net().cpu().eval()
    print('推論:def predict3')
    # 学習済みモデルの重み（NPmodel.pt）を読み込み
    # net.load_state_dict(torch.load('../src/NPmodel.pt', map_location=torch.device('cpu'))) #ここ！！パス？
    net.load_state_dict(torch.load('../src/NPmodel.pt', map_location=torch.device('cpu')))
    
    print('推論:def predict4')
    # 推論
    with torch.no_grad():
        y = net(pred_tensor.to(device).unsqueeze(0))
        print('推論:def predict5')
    # 推論ラベルを取得
    y = torch.argmax(y, dim=-1)
    print(y)
    return y

#------------------
# 結果を整理
#------------------
#推論したyを整理
def Judge(pred):
    cnt_P = 0
    cnt_N = 0
    cnt_Neu = 0

    # 推論結果のyをtensorからPythonに変換
    # y2 = y.flatten().tolist()   # テンソルを1次元に平坦化してからリストに変換
    y2 = pred.flatten().tolist()   # テンソルを1次元に平坦化してからリストに変換
    print('y2:',len(y2))
    print(y2)

    # 全レビューの結果を確認しカウント
    for res in y2:
        #ラベルを取得し、カウント  
        if res == 0:
            cnt_P = cnt_P + 1
        elif res == 1:
            cnt_N = cnt_N + 1
        elif res == 2:
            cnt_Neu = cnt_Neu + 1
        
    # 総合結果
    all_NP = ''
    if cnt_P > cnt_N:
        all_NP = 'ポジティブ'
    elif cnt_P < cnt_N:
        all_NP = 'ネガティブ'
    elif cnt_P == cnt_N:
        if cnt_P > cnt_Neu:
            all_NP = 'ポジティブ'
        elif cnt_P < cnt_Neu:
            all_NP = '中立'
        elif cnt_P == cnt_Neu:
            all_NP = '中立'
    
    #print('P：',cnt_P,'N：',cnt_N,'Neu：',cnt_Neu)        
    #print('総合判定：',all_NP)

    return cnt_P, cnt_N, cnt_Neu, all_NP


    
# Flask のインスタンスを作成
app = Flask(__name__)   

# WTForms を使い、index.html 側で表示させるフォームを構築します。
class InputForm(Form):
    InputFormTest = StringField('価格コムのレビューページのURLを入力してください',
                    [validators.InputRequired()])

    # HTML 側で表示する submit ボタンの表示
    submit = SubmitField('送信')

#URLにアクセスしたときの挙動
# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def predicts():
    # WTFormsで構築したフォームをインスタンス化
    form = InputForm(request.form)

    # POST---
    # Web ページ内の機能に従って処理を行なう    
    if request.method == 'POST':
        
        # 条件に当てはまる場合
        if form.validate() == False:
            return render_template('index.html', forms=form)
            #return render_template('index.html')
        
        # 条件に当てはまらない場合:推論を実行
        else:

            #------------------
            # スクレイピング
            #------------------
            # URLチェック
            input_URL = request.form['InputFormTest']

            # URLからスクレイピング(kakakuCom.pyから関数の呼び出し)         
            res_URL = scrape_kakaku(input_URL)

            # データフレームの表示(kakakuCom.py から読み込み)
            res_df = res_URL            

            #------------------
            # 推論用レビュー
            #------------------
            #filename = '../価格com口コミ.csv'
            #df_Input =pd.read_csv(filename, encoding='shift-jis')
            df_Input = res_df

            # CSVからレビュー部分を順に取り出し、reviewsに格納
            reviews = []    # review内容
            for review in df_Input['comment']:    
                reviews.append(review)
                #print('全レビュー数：', len(reviews))
                

            corpus_In = []
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
                corpus_In.append(' '.join(words))

            print('[1]corpus:',len(corpus_In))

            #Vectorizer
            vectorizer = CountVectorizer(min_df=1, max_df=1.0)
            vectorizer.fit(corpus_In)
            pred_bow = vectorizer.transform(corpus_In).toarray()
            #Tensor
            pred_tensor = torch.tensor(pred_bow, dtype=torch.float32)
            pred_tensor = pred_tensor[:, :133]  # 学習済みモデルの133次元にスライスする
            print('[2]pred_tensor:',len(pred_tensor))

            # 入力されたreviewに対して推論
            pred = predict(pred_tensor)
            print('[3]y:', pred)
            Res_NP_ = Judge(pred)
            # 個別の変数にアクセスして値を利用
            cnt_P_value = Res_NP_[0]
            cnt_N_value = Res_NP_[1]
            cnt_Neu_value = Res_NP_[2]
            all_NP_value = Res_NP_[3]            
            print('[4]総合判定：',all_NP_value,'P:',cnt_P_value,'N:',cnt_N_value,'Neu:',cnt_Neu_value)            
            return render_template('result.html', Res_NP=all_NP_value, Res_P=cnt_P_value,Res_N=cnt_N_value,Res_Neu=cnt_Neu_value)
        return redirect(request.url)
    
    # GET ---
    # URL から Web ページへのアクセスがあった時の挙動
    elif request.method == 'GET':
        return render_template('index.html',forms=form)
    
#アプリ実行の定義
if __name__ == '__main__':
    app.run(debug=True)