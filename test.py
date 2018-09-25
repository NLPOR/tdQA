from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Merge, LSTM, Dense,Embedding
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from keras import regularizers
from dataHelper import *
import time
import pandas as pd
# train_ques = pd.read_csv('./td_data/label_train_question.txt', sep='\t', encoding='utf-8', names=['ques', 'label'])
# test_ques = pd.read_csv('./td_data/label_test_question.txt', sep='\t', encoding='utf-8', names=['ques', 'label'])
# print(train_ques.head())
# print(Counter(train_ques['label']))
# ques_line = []
# for line in open('./td_data/lexer_ques.txt'):
#     line = json.loads(line)
#     print(line)
#     text = ''
#     for word in line['items']:
#         text = text + word['item'] + '_'
#     ques_line.append(text)
#
#
# ques_line = []
# for line in open('./td_data/lexer_ques_test.txt'):
#     line = json.loads(line)
#     print(line)
#     text = ''
#     for word in line['items']:
#         text = text + word['item'] + '_'
#     ques_line.append(text)
# test_ques['ques'] = pd.Series(ques_line)
# test_ques.to_csv('./td_data/label_test_question.txt', sep= '\t', index=None, header=None, encoding='utf-8')
# print(test_ques.head())
# qa_text['ques'] = qa_text['ques'].apply(lambda x: ''.join(x.split('_')))
# qa_text['ques'] = qa_text['ques'].apply(lambda x: nlp.lexer(x))
# qa_text.to_csv('1.txt',sep=' ', encoding= 'utf-8', index=None)
# fw = open('./word2vectModel_200/words.txt', encoding='utf-8').readlines()
# nlp = NLP()
# print(nlp.lexer('在这一年周武王建立了周朝,都城在镐京,史称西周。12:40，北京这个地方很不错，67分，一月十二号，'.replace('_', '')))
from tensorflow.contrib import learn
import numpy as np
max_document_length = 4
x_text =[
    '我 爱你 中国',
    '我 也 爱你'
]
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
vocab_processor.fit(x_text)
print(next(vocab_processor.transform(['我 也 爱你'])).tolist())
x = np.array(list(vocab_processor.fit_transform(x_text)))
print(x)