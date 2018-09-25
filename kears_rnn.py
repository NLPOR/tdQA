import re
import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import classification_report
from collections import OrderedDict
from keras.layers import Input, Embedding, LSTM, Dense, Dropout,Convolution1D,Flatten,MaxPooling1D
from keras.models import Model
from keras.models import model_from_json
from keras.preprocessing import sequence
from keras import regularizers
from keras.utils import plot_model
# from nltk.corpus import stopwords
from collections import Counter
st = PorterStemmer()
def clean_str(s):
    """Clean sentence"""
    #数据预处理
    letters_only = re.sub("[^a-zA-Z]", " ", s)
    words = letters_only.lower().split()
    # s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
    # s = re.sub(r"\'s", " \'s", s)
    # s = re.sub(r"\'ve", " \'ve", s)
    # s = re.sub(r"n\'t", " n\'t", s)
    # s = re.sub(r"\'re", " \'re", s)
    # s = re.sub(r"\'d", " \'d", s)
    # s = re.sub(r"\'ll", " \'ll", s)
    # s = re.sub(r",", " , ", s)
    # s = re.sub(r"!", " ! ", s)
    # s = re.sub(r"\(", " \( ", s)
    # s = re.sub(r"\)", " \) ", s)
    # s = re.sub(r"\?", " \? ", s)
    # s = re.sub(r"\s{2,}", " ", s)
    # s = re.sub(r'\S*(x{2,}|X{2,})\S*',"xxx", s)
    # s = re.sub(r'[^\x00-\x7F]+', "", s)
    # stops = set(stopwords.words("english"))   #if w not in stops
    return [st.stem(w) for w in words ]
def get_submission(predictions):
	test_label = pd.read_csv('./data/test.tsv', sep='\t',
							 names=['id', 'labels'], header=None)
	test_id = test_label['id']
	predictions = pd.DataFrame({'labels':predictions})
	label_map = {0.0:"amber", 1.0:"crisis", 2.0:"green", 3.0:"red"}
	predictions = predictions['labels'].map(label_map)
	result = pd.concat([test_id, pd.Series(predictions)], axis=1)
	result.to_csv('submisson1.tsv', sep='\t', index=None, header=None)
	return predictions


def load_data_and_labels(filename="./data/clean_train_data.csv"):
    #加载训练数据
    """Load sentences and labels"""
    df = pd.read_csv(filename,encoding='gbk')                                   #增加训练样本数据
    # df = pd.read_csv(filename,encoding='gbk')          #只使用现有数据
    selected = ['labels','body']
    #得到labels标签的one-hot表示
    labels = sorted(list(set(df[selected[0]].tolist())))
    one_hot = np.zeros((len(labels),len(labels)),int)
    np.fill_diagonal(one_hot,1)
    label_dict = dict(zip(labels, one_hot))
    #得到x 和 y的示例数据
    x_raw = df[selected[1]].apply(clean_str).tolist()
    print(x_raw)
    y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
    return x_raw, y_raw, df, labels

def loadtest(filename="./data/clean_test_data.csv"):
    #加载训练数据
    """Load sentences and labels"""
    df = pd.read_csv(filename, encoding='gbk')
    df.body = df.body.fillna(method='pad')
    selected = ['body']
    #得到x 和 y的示例数据
    x_raw = df[selected[0]].apply(clean_str).tolist()
    return x_raw

def Word2id(x_raw):
    # 词典
    word2id = OrderedDict()
    word2id["not_in_voc"] = 0
    doc2id = []
    index = 1
    for raw in x_raw:
        doc=[]
        for word in raw:#这里key-value是反的
            if word in word2id:     #pyhton3
            # if word2id.has_key(word):  #python2
                doc.append(word2id[word])
            else:
                word2id[word] = index
                doc.append(index)
                index+=1
        doc2id.append(doc)
    return word2id,doc2id

def train_rnn_model():
    x_raw, y_raw, df, labels = load_data_and_labels()
    Counter(labels)
    word2id,doc2id = Word2id(x_raw)
    print (max([len(doc) for doc in doc2id]))
    doc2id = sequence.pad_sequences(doc2id, maxlen=512,padding='pre',truncating='post')
    #设计模型
    word_size = 64
    maxlen = 512

    input1 = Input(shape=(maxlen,), dtype='int32', name='input1')
    # this embedding layer will encode the input sequence
    # into a sequence of dense 512-dimensional vectors.
    x = Embedding(output_dim=word_size, input_dim=len(word2id), input_length=maxlen)(input1)# 字典长度待定
    # a LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    f = LSTM(64)(x)
    dr = Dropout(0.5)(f)
    # x = Dense(32, activation='relu')(dr)
    # dr = Dropout(0.3)(x)
    main = Dense(4, activation='softmax', name='main',
                 kernel_regularizer=regularizers.l2(0.01),
                 bias_regularizer=regularizers.l1(0.01),
                 activity_regularizer=regularizers.l1_l2(0.01,0.01)
                 )(dr)
    model = Model(input=input1, output=main)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file='model.png')
    print (np.array(doc2id).shape)
    print (np.array(y_raw).shape)
    # ,class_weight={'0':1,'1':8,'2':8,'3':8}
    model.fit(np.array(doc2id), np.array(y_raw), batch_size=64,epochs=15 ,shuffle=True, class_weight = {0.0: 0.15, 1.0: 0.45, 2.0: 0.15, 3.0: 0.25})
    # class_weight = {0.0: 0.15, 1.0: 0.45, 2.0: 0.15, 3.0: 0.25},
    json_string = model.to_json()
    open('./models/my_model_architecture3.json','w').write(json_string)
    model.save_weights('./models/my_model_weights3.h5')

def loadmodel():
    x_raw1, y_raw1, df1, labels1 = load_data_and_labels()
    x_raw = loadtest(filename="./data/clean_test_data.csv")
    wordid,doc2id= Word2id(x_raw1)
    label_map = {0.0:"amber", 1.0:"crisis", 2.0:"green", 3.0:"red"}
    doc2id = []
    index = 1
    for raw in x_raw:
        doc=[]
        for word in raw:#这里key-value是反的
            if word in wordid:
                doc.append(wordid[word])
            else:
                doc.append(0)
                index+=1
        doc2id.append(doc)
    print (max([len(doc) for doc in doc2id]))
    doc2id = sequence.pad_sequences(doc2id, maxlen=512,padding='pre',truncating='post')
    model = model_from_json(open('./models/my_model_architecture3.json').read())
    model.load_weights('./models/my_model_weights3.h5')

    y_pre = model.predict(doc2id)
    y_pre = pd.DataFrame({'labels':[np.argmax(i) for i in y_pre]})
    predictions = y_pre['labels'].map(label_map)

    test_label = pd.read_csv(r'./data/test.tsv', sep='\t', names=['id', 'labels'], header=None)
    test_id = test_label['id']
    # result = pd.concat([test_id, pd.Series(predictions)], axis=1)
    # result.to_csv('./results/submision8.tsv', sep='\t', index=None, header=None)
    print(predictions)
    # print(classification_report([np.argmax(i)for i in y_raw1],y_pre))
    print(classification_report(test_label['labels'],predictions,digits=3))

def concat_data():
    train_df = pd.read_csv("./data/full_sample.csv",encoding='gbk')

    #增加一些预测得到的crisis和red样本
    ex_df = pd.read_csv("./data/ex_data.csv")

    #拼接测试集
    # test_df = pd.read_csv("./data/test_data.csv")
    # test_label = pd.read_csv(r'./data/test.tsv', sep='\t',names=['id', 'labels'], header=None)
    # test_df['labels'] = test_label['labels']
    df = pd.concat([train_df,ex_df],ignore_index=True)
    df['body'] = df['body'].fillna(method='pad')
    return df

if __name__ == "__main__":
    train_rnn_model()
    loadmodel()




