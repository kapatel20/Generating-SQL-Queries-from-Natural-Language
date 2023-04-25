import pandas as pd
import numpy as np
import sklearn
import sklearn.preprocessing
from sklearn.metrics import f1_score, accuracy_score
import tensorflow as tf
import gzip
import json
import re
import copy
from ast import literal_eval
from stanfordcorenlp import StanfordCoreNLP
print("Modules imported.\n")

# nlp = StanfordCoreNLP('http://localhost', port=9000)

fptr = open('../data/WikiSQL/data/train.jsonl', 'r')
queries = fptr.readlines()

dev_fptr = open('../data/WikiSQL/data/dev.jsonl', 'r')
dev_queries = dev_fptr.readlines()

print("Extracting Data...")


trainTabId = []
trainQuestion = []
trainSQL = []
trainSel = []
trainCon = []
trainAgg = []
for i, query in enumerate(queries):
    q = json.loads(query)
    trainTabId.append(q["table_id"])
    trainQuestion.append(q["question"])
    trainSQL.append(q["sql"])
    trainSel.append(q["sql"]["sel"])
    trainCon.append(q["sql"]["conds"])
    trainAgg.append(q["sql"]["agg"])

devTabId = []
devQuestion = []
devSQL = []
devSel = []
devCon = []
devAgg = []
for i, query in enumerate(dev_queries):
    q = json.loads(query)
    devTabId.append(q["table_id"])
    devQuestion.append(q["question"])
    devSQL.append(q["sql"])
    devSel.append(q["sql"]["sel"])
    devCon.append(q["sql"]["conds"])
    devAgg.append(q["sql"]["agg"])

print("Data Extracted.")

def remove_puctuations(text):
    text = re.sub(r'[?]', '', text)
    return text

trainQuestion = np.array([j.lower() for j in trainQuestion])
devQuestion = np.array([j.lower() for j in devQuestion])

ohe = sklearn.preprocessing.OneHotEncoder()
y_train = ohe.fit_transform(np.array(trainAgg).reshape(-1, 1)).toarray()

y_val = ohe.transform(np.array(devAgg).reshape(-1, 1)).toarray()

tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="unk")
tokenizer.fit_on_texts(trainQuestion)
X_train = tokenizer.texts_to_sequences(trainQuestion)
X_val = tokenizer.texts_to_sequences(devQuestion)

max_len = max([len(x) for x in X_train])
print(max_len)
X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(X_train,maxlen=max_len,padding="post", truncating="post")
X_val_pad = tf.keras.preprocessing.sequence.pad_sequences(X_val, maxlen=max_len, padding="post", truncating="post")
print(len(X_train_pad))
gloveEmbeddings = {}
with open('glove.twitter.27B.50d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        embedding = np.asarray(values[1:],dtype='float32')
        gloveEmbeddings[word] = embedding


word_index = tokenizer.word_index
wordEmbedding = np.zeros((len(word_index)+1, 50))

for word,i in word_index.items():
    if(word.lower() in gloveEmbeddings):
        wordEmbedding[i] = gloveEmbeddings[word.lower()]
    else:
        wordEmbedding[i] = gloveEmbeddings['unk']

wordEmbedding = np.array(wordEmbedding)

aggMapping = {'':0,
              'MAX':1,
              'MIN':2,
              'COUNT':3,
              'SUM':4,
              'AVG':5}

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim = len(word_index)+1,
        output_dim = 50,
        weights = [wordEmbedding],
        input_length=48,
        trainable=True
    ),
    # tf.keras.layers.Bidirectional(
    #     tf.keras.layers.LSTM(128)
    # ),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(
        units=128,
        activation="tanh",
        use_bias=True,
    ),
    tf.keras.layers.Dense(
        units=64,
        activation="tanh",
        use_bias=True,
    ),
    tf.keras.layers.Dropout(0.35),
    tf.keras.layers.Dense(
        len(aggMapping),
        activation="softmax"
    )
])
print(model.summary())
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    loss='categorical_crossentropy', 
    optimizer=optimizer, 
    metrics=['accuracy']
)
print(X_val_pad.shape)
print(y_val.shape)
print(X_train_pad.shape)
print(y_train.shape)

checkpoint_path = 'modelsAgg/agg_tf_lstm.h5'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)


history = model.fit(
    X_train_pad, y_train, 
    epochs=20,
    batch_size=64, 
    validation_data=(X_val_pad, y_val),
    callbacks=[checkpoint_callback]
    )

# history_df = pd.DataFrame(history)
# history_df.to_csv("modelsAgg/history.csv")


