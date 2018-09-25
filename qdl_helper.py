import os
import pandas as pd
import json
import time
import logging
import td_data_helper
import numpy as np
import tensorflow as tf
from qdl_model import TextCNN
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from collections import Counter
from sklearn.metrics import classification_report
logging.getLogger().setLevel(logging.INFO)
def batch_iter(data, batch_size, num_epochs, shuffle = True):
    """Iterate the data batch by btach"""
    #输入数据[x,y], 每一次epoch的batch的大小，多少步num_epochs
    # data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size/batch_size) + 1

    for epoch in range(num_epochs):
        # if shuffle:
        #     shuffle_indices = np.random.permutation(np.arange(data_size))
        #     shuffle_data = data[shuffle_indices]
        # else:
        #     shuffle_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num+1)*batch_size, data_size)
            yield data[start_index:end_index]

def padding(text, max_len ):
    '''
    对问题和答案进行padding操作使得问题和答案长度保持一致
    :param in_file:   输入文件
    :param out_file:  padding后保存的文件
    :param max_len:   padding的最大长度
    :return:
    '''
    try:
        text = text.split('_')
    except:
        text = ' '
    if max_len < len(text):  #问题长度大于最大长度
        ques = '_'.join(text[:max_len])
    else:
        ques = '_'.join(text) + '_' + "a_" * (max_len-len(text))
    return ques

def connect_qa():
    df = pd.read_csv('./td_data/train_data_complete.txt', header=None, sep=' ',
                           encoding='utf-8',names=['label', 'q_id', 'question', 'anwser', 'anwser_id'])
    df2 = pd.read_csv('./td_data/test_data.txt', header=None, sep=' ',
                           encoding='utf-8',names=['label', 'q_id', 'question', 'anwser', 'anwser_id'])
    train_qa = pd.DataFrame()
    train_qa['q_a'] = df['question'] + '_' + df['anwser']
    train_qa['label'] = df['label']
    train_qa.to_csv('./td_data/train_data_qa.txt', encoding='utf-8',header=None, index=None,sep='\t')

    test_qa = pd.DataFrame()
    test_qa['q_a'] = df2['question'] + '_' + df2['anwser']
    test_qa['label'] = df2['label']
    test_qa.to_csv('./td_data/test_data_qa.txt', encoding='utf-8',header=None, index=None,sep='\t')
def loadTrainData(file = './td_data/train_data_qa.txt'):
    x_train_ids = []
    x_train_words = []
    y_train = []
    word2id = OrderedDict()
    items_index = 0
    with open(file,encoding='utf8') as f:
        for line in f:
            question_label = line.split('\t')
            words = question_label[0].split('_')
            label = question_label[1]
            temp = []
            for word in words:
                # print(word2id.get(word))
                if word2id.get(word):# 已有的话，只是当前文档追加
                    temp.append(word2id[word])
                else:  # 没有的话，要更新vocabulary中的单词词典及wordidmap
                    word2id[word] = items_index
                    temp.append(items_index)
                    items_index += 1
            x_train_ids.append(temp)
            x_train_words.append(words)
            y_train.append(label)
    return x_train_ids,x_train_words,y_train,items_index,word2id

def loadTestData(items_index,word2id,file = './td_data/test_data_qa.txt'):
    x_train_ids = []
    x_train_words = []
    y_train = []
    with open(file,encoding='utf8') as f:
        for line in f:
            question_label = line.split('\t')
            words = question_label[0].split('_')
            label = question_label[1]
            temp = []
            for word in words:
                # print(word2id.get(word))
                if word2id.get(word):# 已有的话，只是当前文档追加
                    temp.append(word2id[word])
                else:  # 没有的话，要更新vocabulary中的单词词典及wordidmap
                    word2id[word] = items_index
                    temp.append(items_index)
                    items_index += 1
            x_train_ids.append(temp)
            x_train_words.append(words)
            y_train.append(label)
    return x_train_ids,x_train_words,y_train,items_index,word2id

def loadWord2Vec(word2id,file = 'word2vectModel_200/wiki.zh.seg_200d.vec'):
    with open(file,encoding="utf8") as f:
        index = 0
        for line in f:
            if index == 0:
                index += 1
                pass
            else:
                index += 1
                feature =  line.split()
                if word2id.get(feature[0]):
                    word2id[feature[0]] = str(word2id[feature[0]]) + '\t'+' '.join(feature[1:])
                else:
                    pass
    return word2id

def getXtrainAndXtestFeatures(x_train_words,x_test_words,word2idAndFeature):
    x_train_feature = np.zeros((len(x_train_words),1024,200),dtype="float32")
    x_test_feature = np.zeros((len(x_test_words),1024,200),dtype="float32")
    index = 0
    for words in x_train_words:
        temp = np.zeros((1024,200),dtype="float32")
        j = 0
        for word in words:
            if word =='a' or isinstance(word2idAndFeature.get(word), int):
                word_feature = np.zeros((200),dtype="float32")
                temp[j] = word_feature
            else:
                word_feature = np.array(word2idAndFeature.get(word).split('\t')[1].split(' '),dtype="float32")
                temp[j] = word_feature
            j += 1
        x_train_feature[index] = temp
        index += 1
    index = 0
    for words in x_test_words:
        temp = np.zeros((2000,200),dtype="float32")
        j = 0
        for word in words:
            if word =='a' or isinstance(word2idAndFeature.get(word), int):
                word_feature = np.zeros((200),dtype="float32")
                temp[j] = word_feature
            else:
                word_feature = np.array(word2idAndFeature.get(word).split('\t')[1].split(' '),dtype="float32")
                temp[j] = word_feature
            j += 1
        x_test_feature[index] = temp
        index += 1
    return x_train_feature,x_test_feature

def main():
    x_train_ids, x_train_words, y_train, items_index, word2id = loadTrainData()
    x_test_ids, x_test_words, y_test, items_index, word2id = loadTestData(items_index, word2id,
                                                                              file='./td_data/test_data_qa.txt')
    max_len = max([max([len(ids) for ids in x_train_ids]),max([len(ids) for ids in x_train_ids])])
    # word2idAndFeature = loadWord2Vec(word2id)
    # x_train_feature,x_test_feature = getXtrainAndXtestFeatures(x_train_words,x_test_words,word2idAndFeature)
    y_test = pd.read_csv('./td_data/submit_sample.txt', sep=',', header=None,names=['id', 'label'])['label']
    return x_train_ids,y_train,x_test_ids,y_test,items_index,max_len

# main()

