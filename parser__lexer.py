#encoding:utf-8
from td_data_helper import NLP
import pandas as pd
import numpy as np
import json

#第一步先断句
def cut_sentences(sentence):
    if not isinstance(sentence, str):
        sentence = str(sentence)
    puns = ['。', '!', '?']
    tmp = []
    i = 0
    for ch in sentence:
        tmp.append(ch)
        if ch in puns:
            i+=1
            print('第{}个句子。'.format(i))
            yield ''.join(tmp)
            tmp = []
    yield ''.join(tmp)


def get_sentences(text):
    sentences = []
    for i in cut_sentences(text):
        sentences.append(i)
    print(sentences)
    return sentences

def anwser_sentences():
    '''
    对答案进行分句处理
    :return:
    '''
    f_train = open('./td_data/train_anwser.txt', 'w', encoding='utf-8')
    f_test = open('./td_data/test_anwser.txt', 'w', encoding='utf-8')
    with open('./td_data/train_data.txt',encoding='utf-8') as f:
        for line in f:
            s = get_sentences(line.split(' ')[3])
            f_train.write(json.dumps(s))
            f_train.write('\n')

    with open('./td_data/test_data.txt',encoding='utf-8') as f:
        for line in f:
            s = get_sentences(line.split(' ')[3])
            f_test.write(json.dumps(s))
            f_test.write('\n')


def ques_list_lexer(file = './td_data/test_data.txt'):
    '''
    #得到问题，和对应每个答案的词法分析的列表
    :param file:
    :return:
    '''

    nlp = NLP()
    trainList = []
    for line in open(file, encoding='utf-8'):
        trainList.append(line.strip())
    x_train_1 = []          #测试集中的问题
    x_train_2 = []          #测试集中的候选答案
    ques_list = []          #最后的问题列表
    for i in range(0, len(trainList)):
        items = trainList[i].split(' ')
        ques = trainList[i].split(' ')[2]            #当前问题
        try:
            if ques == trainList[i+1].split(' ')[2]:     #如果是同一个问题
                # print('同一个问题')
                x_train_1.append(items[2])
                try:
                    x_train_2.append(nlp.lexer(items[3].replace('_', ''))['items'])
                except:
                    x_train_2.append(nlp.lexer(items[3]))
            else:
                # print('不同问题')
                x_train_1.append(items[2])
                try:
                    x_train_2.append(nlp.lexer(items[3].replace('_', ''))['items'])
                except:
                    x_train_2.append(nlp.lexer(items[3]))
                ques_list.append([x_train_1, x_train_2])
                x_train_1 = []  # 测试集中的问题
                x_train_2 = []  # 测试集中的候选答案

        except:
            x_train_1.append(items[2])
            x_train_2.append(nlp.lexer(items[3].replace('_', ''))['items'])

            ques_list.append([x_train_1, x_train_2])
    return ques_list

def load_anwser():
    nlp = NLP()
    # f_train = open('./td_data/train_anwser_parser.txt', 'w', encoding='utf-8')
    # f_test = open('./td_data/test_anwser_parser.txt', 'w', encoding='utf-8')
    # with open('./td_data/train_anwser.txt',encoding='utf-8') as f:
    #     i = 0
    #     for line in f:
    #         x= []
    #         sentences = json.loads(line)
    #         sentences = [line.replace('_', '') for line in sentences]
    #         for line in sentences:
    #             sen = nlp.depParser(line)
    #             if 'items' in sen:
    #                 x.append(sen['items'])
    #         f_train.write(json.dumps(x))
    #         f_train.write('\n')
    #
    #
    #
    # with open('./td_data/test_anwser.txt',encoding='utf-8') as f:
    #     for line in f:
    #         x = []
    #         sentences = json.loads(line)
    #         sentences = [line.replace('_', '') for line in sentences]
    #         for line in sentences:
    #             sen = nlp.depParser(line)
    #             if 'items' in sen:
    #                 x.append(sen['items'])
    #         f_test.write(json.dumps(x))
    #         f_test.write('\n')


    f_train = open('./td_data/train_anwser_lexer.txt', 'w', encoding='utf-8')
    f_test = open('./td_data/test_anwser_lexer.txt', 'w', encoding='utf-8')

    with open('./td_data/test_data.txt',encoding='utf-8') as f:
        i = 0
        for line in f:
            i+=1
            print(i)
            s = ''.join(line.split(' ')[3].split('_'))
            try:
                f_test.write(json.dumps(nlp.lexer(s)['items']))
                f_test.write('\n')
            except:
                f_test.write(json.dumps(nlp.lexer(s)))
                f_test.write('\n')

    with open('./td_data/train_data.txt',encoding='utf-8') as f:
        for line in f:
            s = s = ''.join(line.split(' ')[3].split('_'))
            try:
                f_train.write(json.dumps(nlp.lexer(s)['items']))
                f_train.write('\n')
            except:
                print()
                f_train.write(json.dumps(nlp.lexer(s)))
                f_train.write('\n')


# anwser_sentences()    #
# load_anwser()
# ques_list_lexer() #得到问题，和对应每个答案的词法分析的列表

