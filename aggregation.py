# -*- coding: utf-8 -*-
"""Aggregation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AQd5x6Lc5mcpopC3K0WKOjqiZK6zrw-i
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
import torch.optim.lr_scheduler as lr_scheduler
import copy
import gzip
import json
import re
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def readLines(path):
  fptr = open(path, 'r')
  lines = fptr.readlines()
  return lines

trainQuery = readLines('/content/drive/MyDrive/Project/WikiSQL/data/train.jsonl')
devQuery = readLines('/content/drive/MyDrive/Project/WikiSQL/data/dev.jsonl')
testQuery = readLines('/content/drive/MyDrive/Project/WikiSQL/data/test.jsonl')

trainTables = readLines('/content/drive/MyDrive/Project/WikiSQL/data/train.tables.jsonl')
devTables = readLines('/content/drive/MyDrive/Project/WikiSQL/data/dev.tables.jsonl')
testTables = readLines('/content/drive/MyDrive/Project/WikiSQL/data/test.tables.jsonl')

def prepareQueryData(queryData):
    tabId = []
    question = []
    SQL = []
    # selectCol = []
    whereCondition = []
    aggOperator = []
    whereCol = []
    whereOp = []
    whereVal = []
    for i, query in enumerate(queryData):
        q = json.loads(query)
        if len(q["sql"]["conds"]) > 0:
            tabId.append(q["table_id"])
            question.append(q["question"])
            SQL.append(q["sql"])
            # selectCol.append(q["sql"]["sel"])
            # whereCondition.append(q["sql"]["conds"])
            # print(q["sql"]["conds"])
            # whereCol.append(q["sql"]["conds"][0][0])
            # whereOp.append(q["sql"]["conds"][0][1])
            # whereVal.append(q["sql"]["conds"][0][2])
            aggOperator.append(q["sql"]["agg"])
    return tabId, question, SQL, aggOperator

def prepareTableData(tableData, tableID, selectCol):
    columnName = []
    columnDict = {}
    allColumnList = []
    targetColumn = []

    for i, data in enumerate(tableData):
        q = json.loads(data)
        columnDict[q['id']] = q['header']

    for id in tableID:
        columnName.append(columnDict[id])
        allColumnList.extend(columnDict[id])

    for i, col in enumerate(selectCol):
        targetColumn.append(columnName[i][col])
    
    return columnName, allColumnList, targetColumn

def prepareData(lines):
    sentences = []
    for line in lines:
        lst = line.split(' ')
        sentences.append(lst)
    return sentences

def remove_puctuations(temp):
    # if len(text)<=1:
    #   return text
    text = temp
    # text = re.sub(r'[?]',' ', text)
    text = re.sub(r'[^A-Za-z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    if text==" " or text=="":
        return temp
    return text

def createMapping(sentences):
    vocabMapping = {}
    vocabMapping['<pad>'] = 0
    vocabMapping['<unk>'] = 1
    i=2
    vocab = []
    for words in sentences:
        vocab.extend(words)

    vocabDict = {}
    for v in vocab:
        if v.lower() in vocabDict:
            vocabDict[v.lower()]+=1
        else:
            vocabDict[v.lower()]=1

    vocab = list(vocabDict.keys())
    for word in vocab:
        vocabMapping[word.lower()] = i
        i+=1
    return vocabMapping

def prepareSentences(lines, vocabMapping):
    sentences = []

    for line in lines:
      sentence = []

      for word in line:
        if(word.lower() in vocabMapping):
            sentence.append(vocabMapping[word.lower()])
        else:
            sentence.append(vocabMapping['<unk>'])
      sentences.append(sentence)

    sentenceLengths = [len(x) for x in sentences]
    maxLen = max(sentenceLengths)
    listS = []

    for sentence in sentences:
        padLength = maxLen - len(sentence)
        padLength = max(0, padLength)
        sentence.extend([0 for i in range(padLength)])
        # print(sentence)
        listS.append(sentence)

    return listS, sentenceLengths

def createPreTrainedEmbed(vocabMapping):
    gloveEmbeddings = {}
    with open('/content/drive/MyDrive/Project/WikiSQL/glove.twitter.27B.50d.txt') as f:
        for line in f:
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:],dtype='float32')
            gloveEmbeddings[word] = embedding

    wordEmbedding = np.zeros((len(vocabMapping), 50))
    for word in vocabMapping:
        if(word.lower() in gloveEmbeddings):
            wordEmbedding[vocabMapping[word]] = gloveEmbeddings[word.lower()]
        else:
            wordEmbedding[vocabMapping[word]] = gloveEmbeddings['unk']
    wordEmbeds = torch.Tensor(wordEmbedding)
    return wordEmbeds

trainTabId, trainQuestion, trainSQL, trainAgg = prepareQueryData(trainQuery)
devTabId, devQuestion, devSQL, devAgg = prepareQueryData(devQuery)
testTabId, testQuestion, testSQL, testAgg = prepareQueryData(testQuery)

trainQuestion = [remove_puctuations(x) for x in trainQuestion]
devQuestion = [remove_puctuations(x) for x in devQuestion]
testQuestion = [remove_puctuations(x) for x in testQuestion]
trainQuestions = prepareData(trainQuestion)
devQuestions = prepareData(devQuestion)
testQuestions = prepareData(testQuestion)
vocabMapping = createMapping(trainQuestions)
trainQuestions, questionLengths = prepareSentences(trainQuestions, vocabMapping)
devQuestions, devLength = prepareSentences(devQuestions, vocabMapping)
testQuestions, testLength = prepareSentences(testQuestions, vocabMapping)

traindataset = TensorDataset(torch.tensor(trainQuestions), torch.tensor(trainAgg), torch.tensor(questionLengths))
trainLoader = DataLoader(traindataset, batch_size= 64, shuffle=True)

devdataset = TensorDataset(torch.tensor(devQuestions), torch.tensor(devAgg), torch.tensor(devLength))
devLoader = DataLoader(devdataset, batch_size= 1, shuffle=True)

testdataset = TensorDataset(torch.tensor(testQuestions), torch.tensor(testAgg), torch.tensor(testLength))
testLoader = DataLoader(testdataset, batch_size= 1, shuffle=True)

# vocabMapping
wordEmbeds = createPreTrainedEmbed(vocabMapping)



class BiLSTM(nn.Module):
    def __init__(self, embeddingDim=50, lstmHiddenDim=128, linearOutDim=128, vocabSize=0, tagSize=0, pretrainedEmbed=None, freeze=True):
        super(BiLSTM, self).__init__()        
        # self.wordembed = nn.Embedding(vocabSize, embeddingDim)
        self.wordembed = nn.Embedding.from_pretrained(pretrainedEmbed, freeze = freeze)
        self.bilstm = nn.LSTM(input_size = embeddingDim, hidden_size = lstmHiddenDim, bidirectional = True, batch_first = True, num_layers=1)
        self.dropout = nn.Dropout(p=0.1)
        self.linear1 = nn.Linear(2*lstmHiddenDim, linearOutDim)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(linearOutDim, tagSize)
        
    def forward(self, x, xLength):
        x = pack_padded_sequence(x, xLength.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = pad_packed_sequence(x, batch_first=True)
        wordEmbedding = self.wordembed(x)
        out, (h,c) = self.bilstm(wordEmbedding)
        out = out[:,-1,:]
        out = self.dropout(out)
        out = self.tanh(out)
        out = self.linear1(out)
        out = self.dropout(out)
        out = self.tanh(out)
        out = self.linear2(out)
        out = self.dropout(out)
        return out

aggMapping = {'':1,
              'MAX':2,
              'MIN':3,
              'COUNT':4,
              'SUM':5,
              'AVG':6}

def evalModel(model, loader):
    running_loss = 0.0
    lossfunction = nn.CrossEntropyLoss()
    model.eval()
    pred = []
    true = [] 
    with torch.no_grad():
        for i, (Xbatch ,Ybatch, sentenceLen) in enumerate(loader):
            Xbatch = Xbatch.to(device)
            Ybatch = Ybatch.to(device)
            ypred = model(Xbatch.long(), sentenceLen)
            pred += (torch.argmax(ypred, dim=1).float().tolist())
            true += (Ybatch.view(-1).tolist())
            loss = lossfunction(ypred.to(device), Ybatch.view(-1).type(torch.LongTensor).to(device))
            running_loss += loss.item()*Xbatch.size(0)
    return f1_score(true, pred, average="macro"), accuracy_score(true, pred)

def trainModel(trainLoader, model, epoch):
    num_epochs = epoch
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    lossfunction = nn.CrossEntropyLoss()

    best_model = None
    best_epoch = None
    best_f1 = -1

    for epoch in range(num_epochs):
        model.train()
        train_loss=0
        for i, (Xbatch ,Ybatch, sentenceLen) in enumerate(trainLoader):
            Xbatch = Xbatch.to(device)
            Ybatch = Ybatch.to(device)
            optimizer.zero_grad()
            ypred = model(Xbatch.long(), sentenceLen)
            loss = lossfunction(ypred.to(device), Ybatch.type(torch.LongTensor).to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*Xbatch.size(0)
           

        temp_val_f1, temp_val_acc = evalModel(model, devLoader)
        if temp_val_f1 > best_f1:
          best_f1 = temp_val_f1
          best_model = copy.deepcopy(model)
          best_epoch = epoch+1
        train_loss = train_loss/len(trainLoader.dataset)
        print('Epoch: {}\t --->\tTraining Loss: {:.6f} \t--->\tValidation F1 : {:.6} \t--->Validation Accuracy :{:.6}'.format(epoch+1, train_loss,temp_val_f1,temp_val_acc))
    return model, best_model

model = BiLSTM(vocabSize = len(vocabMapping), tagSize = len(aggMapping), pretrainedEmbed=wordEmbeds, freeze=True)
model = model.to(device)

model, best_model = trainModel(trainLoader, model, epoch=20)

val_f1, val_acc = evalModel(best_model, devLoader)

print("Dev Data : F1 Score : ", val_f1)
print(" Dev Data Accuracy score : ", val_acc)

test_val_f1, test_val_acc = evalModel(best_model, testLoader)

print("Test Data : F1 Score : ", test_val_f1)
print(" Test Data Accuracy score : ", test_val_acc)