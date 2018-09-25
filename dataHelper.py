import numpy as np
from collections import OrderedDict
import pandas as pd
from collections import Counter
def loadTrainData(file = './td_data/pad_tr_ques.txt'):
    x_train_ids = []
    x_train_words = []
    y_train = []
    word2id = OrderedDict()
    items_index = 0
    with open(file,encoding='utf8') as f:
        for line in f:
            question_label = line.split('\t')
            words = question_label[0].rstrip('_').split('_')
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

def loadTestData(items_index,word2id,file = './td_data/pad_te_ques.txt'):
    x_train_ids = []
    x_train_words = []
    y_train = []
    with open(file,encoding='utf8') as f:
        for line in f:
            question_label = line.split('\t')
            words = question_label[0].rstrip('_').split('_')
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
    x_train_feature = np.zeros((len(x_train_words),63,200),dtype="float32")
    x_test_feature = np.zeros((len(x_test_words),63,200),dtype="float32")
    index = 0
    for words in x_train_words:
        temp = np.zeros((63,200),dtype="float32")
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
        temp = np.zeros((63,200),dtype="float32")
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

def get_ques(file = './td_data/test_data.txt'):
    trainList = []
    for line in open(file, encoding='utf-8'):
        trainList.append(line.strip())
    x_train_1 = []          #测试集中的问题
    x_train_2 = []          #测试集中的候选答案
    x_train_3 = []          #测试集中的候选答案
    ques_list = []          #最后的问题列表
    for i in range(0, len(trainList)):
        items = trainList[i].split(' ')
        ques = trainList[i].split(' ')[2]            #当前问题
        try:
            if ques == trainList[i+1].split(' ')[2]:     #如果是同一个问题
                # print('同一个问题')
                x_train_1.append(items[2])
                x_train_2.append(items[3])

            else:
                # print('不同问题')
                x_train_1.append(items[2])
                x_train_2.append(items[3])

                ques_list.append([x_train_1, x_train_2])
                x_train_1 = []  # 测试集中的问题
                x_train_2 = []  # 测试集中的候选答案

        except:
            x_train_1.append(items[2])
            x_train_2.append(items[3])

            ques_list.append([x_train_1, x_train_2])
    return ques_list

def main():
    # ids有点浪费，与后面重复
    x_train_ids,x_train_words,y_train,items_index,word2id = loadTrainData()
    x_test_ids,x_test_words,y_test,items_index,word2id = loadTestData(items_index,word2id,file = './td_data/pad_te_ques2.txt')

    word2idAndFeature = loadWord2Vec(word2id)
    x_train_feature,x_test_feature = getXtrainAndXtestFeatures(x_train_words,x_test_words,word2idAndFeature)
    return x_train_feature,y_train,x_test_feature,y_test

def connect():
    pad_tr_ques = pd.read_csv('./td_data/pad_tr_ques.txt', sep='\t',names=['ques', 'label'], encoding='utf-8')
    pad_te_ques = pd.read_csv('./td_data/pad_te_ques.txt', sep='\t', names=['ques', 'label'], encoding='utf-8')
    for dex in pad_te_ques.index:
        if pad_te_ques.loc[dex, 'label'] == 5 or pad_te_ques.loc[dex, 'label'] == 6:
            pad_te_ques.loc[dex, 'label'] = 4
    for dex in pad_tr_ques.index:
        if pad_tr_ques.loc[dex, 'label'] == 5 or pad_tr_ques.loc[dex, 'label'] == 6:
            pad_tr_ques.loc[dex, 'label'] = 4

    pad_tr_ques.to_csv('./td_data/pad_tr_ques2.txt', index=None, header=None, encoding='utf-8', sep='\t')
    pad_te_ques.to_csv('./td_data/pad_te_ques2.txt', index=None, header=None, encoding='utf-8', sep='\t')
    print(Counter(pad_te_ques['label']))
    print(Counter(pad_tr_ques['label']))

    return


# x_train_feature,y_train,x_test_feature,y_test = main()
# get_ques()

# loadWord2Vec()

