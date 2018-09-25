import json
import numpy as np
import re
import datetime
import jieba
import random
from aip import AipNlp
import pandas as pd
from gensim.models.word2vec import Word2Vec
train_data_file = './td_data/train_data_complete.json'
test_data_file = './td_data/test_data_sample.json'
vocabs = './td_data/vocabs.json '

class NLP(object):

    def __init__(self):
        self.APP_ID = '10936146'
        self.API_KEY = 'nmdtXSQoq3K2MxeOXUpsPxsv'
        self.SECRET_KEY = 'T8IzKYjFhAm7BqjeRC9ICfEDQBGp91fG'
        self.client = AipNlp(self.APP_ID, self.API_KEY, self.SECRET_KEY)
    def depParser(self, text):
        '''
        输入一段文本text，返回text的句法依存结果
        :param text:
        :return:
        '''
        print("句法依存分析后的结果：")
        print(self.client.depParser(text))
        return self.client.depParser(text)
    def lexer(self, text):
        '''
        词法分析,输入一段文本，返回此法分析后的结果
        :return:
        '''
        print("词法分析后的结果：")
        print(self.client.lexer(text))
        return self.client.lexer(text)

    def get_word2vec(self, word):
        word_2vec = self.client.wordEmbedding(word)
        print("词向量：", word_2vec)
        return word_2vec

def get_max_len():
    max_len = 0
    for line in open('./td_data/train_data.txt', encoding='utf-8'):
        items = line.strip().split(' ')
        text = items[3]
        if max_len < len(text.split('_')):
            print(len(text.split('_')))
            print(text.split('_'))
            max_len = len(text.split('_'))
    for line in open('./td_data/test_data.txt', encoding='utf-8'):
        items = line.strip().split(' ')
        text = items[3]
        if max_len < len(text.split('_')):
            max_len = len(text.split('_'))
    return max_len

def padding(in_file, out_file, max_len):
    '''
    对问题和答案进行padding操作使得问题和答案长度保持一致
    :param in_file:  输入文件
    :param out_file:  padding后保存的文件
    :param max_len:   padding的最大长度
    :return:
    '''
    fw = open(out_file, 'w', encoding='utf-8')
    for line in open(in_file, encoding='utf-8'):
        items = line.strip().split(' ')
        ques = items[2]
        ans = items[3]
        ques_list = ques.split('_')
        ans_list = ans.split('_')

        if max_len < len(ques_list):  #问题长度大于最大长度
            ques = '_'.join(ques_list[:max_len])
        else:
            ques = ques + "_" + "a_" * (max_len-len(ques_list))

        if max_len < len(ans_list): #问题长度大于最大长度
            ans = '_'.join(ans_list[:max_len])
        else:
            ans = ans + "_" + "a_" * (max_len-len(ans_list))

        fw.write(' '.join([items[0], items[1], ques, ans, items[4]]))
        fw.write('\n')



    return

def build_vocab():
    '''
    #建立词库表
    :return:
    '''
    code = int(0)               #code为0
    vocab = {}                  #词库字典
    vocab['UNKNOWN'] = code     #词库中没有的词（未登录词）设置为0
    code += 1
    for line in open('./td_data/pd_train_data_complete.txt', encoding='utf-8'):
        items = line.strip().split(' ')
        for i in range(2, 4):
            words = items[i].split('_')
            for word in words:
                if not word in vocab:
                    vocab[word] = code
                    code += 1
    for line in open('./td_data/test_data2.txt', 'r', encoding='utf-8'):
        items = line.strip().split(' ')
        for i in range(2, 4):
            words = items[i].split('_')
            for word in words:
                if not word in vocab:
                    vocab[word] = code
                    code += 1
    print('词库的长度：',len(vocab))
    fw = open(vocabs,'w', encoding='utf-8')
    fw.write(json.dumps(vocab))
    print("词库表保存完成！！！！")
    return vocab

def load_vocabs():
    f = open('./td_data/vocabs.json',encoding='utf-8').read()
    vocab = json.loads(f)
    return vocab

def rand_qa(qalist):
    '''
    从qalist列表中随机选择一个元素
    :param qalist:
    :return:
    '''
    index = random.randint(0, len(qalist) - 1)
    return qalist[index]

def read_alist():
    '''
    读取所有训练集中的候选答案，不管答案是否是正确答案
    :return:
    '''
    alist = []
    for line in open('./td_data/pd_train_data_complete.txt',encoding='utf-8'):
        items = line.strip().split(' ')
        alist.append(items[3])
    print('read_alist done ......')
    return alist

def encode_sent(vocab, string):
    '''
    对所给句子进行编码
    :param vocab:   词表
    :param string:  句子
    :return:
    '''
    x = []
    words = string.split('_')
    for i in range(0, len(words)-1):
        if words[i] in vocab:
            x.append(vocab[words[i]])
        else:
            x.append(vocab['UNKNOWN'])
    return x

def load_data_6(vocab, alist, raw, size):
    '''
    :param vocab:    词库表
    :param alist:    所有的训练数据
    :param raw:      所有的正确答案的数据
    :param size:     选取的数据大小batch_size
    :return:
    '''
    x_train_1 = []      #问题
    x_train_2 = []      #正向答案
    x_train_3 = []      #负向答案
    for i in range(0, size):        #
        items = raw[random.randint(0, len(raw) - 1)]
        nega = rand_qa(alist)
        x_train_1.append(encode_sent(vocab, items[2]))     #问题
        x_train_2.append(encode_sent(vocab, items[3]))     #正向答案
        x_train_3.append(encode_sent(vocab, nega))         #从候选答案中随机选择一个当做是负向答案
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)

def load_vectors():
    '''
    加载词向量，得到词向量的词典
    :return: 返回一个{‘word’：vector}的词典vectors
    '''
    vectors = {}
    w2v_model = Word2Vec.load('./word2vectModel_200/wiki.zh.seg_200d.model')
    w2v_model.wv.save_word2vec_format('./word2vectModel_200/wiki.zh.seg_200d.vec')
    for line in open('./word2vectModel_200/wiki.zh.seg_200d.vec', encoding='utf-8'):
        print(line)
        break
        items = line.strip().split(' ')
        if (len(items) < 101):
            continue
        vec = []
        for i in range(1, 101):
            vec.append(float(items[i]))
        vectors[items[0]] = vec


    return vectors

def load_data_test(testList, vocab):
    '''
    生成所有
    :param testList:    所有的测试数据列表
    :param vocab:       词库列表
    :return:
    '''
    x_train_1 = []          #测试集中的问题
    x_train_2 = []          #测试集中的候选答案
    x_train_3 = []          #测试集中的候选答案
    ques_list = []          #最后的问题列表
    for i in range(0, len(testList)):
        items = testList[i].split(' ')
        ques = testList[i].split(' ')[2]            #当前问题
        try:
            if ques == testList[i+1].split(' ')[2]:     #如果是同一个问题
                # print('同一个问题')
                x_train_1.append(encode_sent(vocab,items[2]))
                x_train_2.append(encode_sent(vocab,items[3]))
                x_train_3.append(encode_sent(vocab,items[3]))
            else:
                # print('不同问题')
                x_train_1.append(encode_sent(vocab,items[2]))
                x_train_2.append(encode_sent(vocab,items[3]))
                x_train_3.append(encode_sent(vocab,items[3]))
                ques_list.append([x_train_1, x_train_2, x_train_3])
                x_train_1 = []  # 测试集中的问题
                x_train_2 = []  # 测试集中的候选答案
                x_train_3 = []  # 测试集中的候选答案
        except:
            x_train_1.append(encode_sent(vocab, items[2]))
            x_train_2.append(encode_sent(vocab, items[3]))
            x_train_3.append(encode_sent(vocab, items[3]))
            ques_list.append([x_train_1, x_train_2, x_train_3])
    return ques_list

def get_test_batch(ques_test, batch_size):
    test_size = len(ques_test)
    num_batches_per_epoch = int(len(ques_test)/batch_size) + 1
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, test_size)
        yield ques_test[start_index:end_index]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def get_batches(arr, n_seqs, n_steps):
    '''
    对已有的数组进行mini-batch分割

    arr: 待分割的数组
    n_seqs: 一个batch中序列个数
    n_steps: 单个序列包含的字符数
    '''
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    # 这里我们仅保留完整的batch，对于不能整出的部分进行舍弃
    arr = arr[:batch_size * n_batches]

    # 重塑
    arr = arr.reshape((n_seqs, -1))

    for n in range(0, arr.shape[1], n_steps):
        # inputs
        x = arr[:, n:n + n_steps]
        # targets
        y = np.zeros_like(x)  # 得到一个与x的shape大小一样的array
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y

def load_test_and_vectors():
    testList = []
    for line in open('./td_data/test_data2.txt', encoding='utf-8'):
        testList.append(line.strip())
    # vectors = load_vectors()
    print('test_data load done')
    return testList

def read_raw(data_file = './td_data/rw_train_data_complete.txt'):   #读取正确答案的数据
    raw = []
    qa_list = []    #问题答案列表， [q, a+, a-]
    for line in open(data_file,'r', encoding='utf-8'):
        items = line.strip().split(' ')
        raw.append(items)
        qa_list.append(items[2])
    return raw

def clean_str(text):
    # text = re.sub('~!@#$%^&*\(\)', '', text)
    re_noise = re.compile('[^0-9a-zA-Z.,：?。！《》*÷+\u4e00-\u9f5a]')
    text = re.sub('_', '_', text)
    text = re_noise.sub('', text)
    return text

def trans_data(in_file, out_file):

    string = open(in_file, 'r', encoding='utf-8').read()
    s_time = datetime.datetime.now()
    print('{}数据格式转换。。。。。'.format(in_file))
    print('开始时间：', s_time)
    data = json.loads(string)
    fw = open(out_file, 'w', encoding='utf-8')
    for question in data:
        ques_id = question['item_id']       #问题id
        ques = '_'.join(jieba.cut(clean_str(question['question'])))         #问题
        all_answer = question['passages']  #问题的所有候选答案
        for message in all_answer:
            answer = '_'.join(jieba.cut(clean_str(message['content'])))
            answer_id = message['passage_id']
            if 'label' in message.keys():
                label = message['label']
                fw.write(' '.join([str(label), str(ques_id), ques.strip(), answer.strip(), str(answer_id)]))
                fw.write('\n')
            else:
                fw.write(' '.join(['un', str(ques_id), ques.strip(), answer.strip(), str(answer_id)]))
                fw.write('\n')
    e_time = datetime.datetime.now()
    print('数据转换结束,结束时间为：', e_time)
    print('共耗时间：', e_time - s_time)

def right_answer():
    fw = open('./td_data/rw_train_data_complete.txt', 'w', encoding='utf-8')
    for line in open('./td_data/pd_train_data_complete.txt', 'r', encoding='utf-8'):
        if int(line[0]) == 1:
            fw.write(line)

    fw.close()

def get_ques():
    trainList = []
    # vocab = load_vocabs()
    for line in open('./td_data/train_data_complete.txt', encoding='utf-8'):
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
                x_train_3.append(items[3])
            else:
                # print('不同问题')
                x_train_1.append(items[2])
                x_train_2.append(items[3])
                x_train_3.append(items[3])
                ques_list.append([x_train_1, x_train_2, x_train_3])
                x_train_1 = []  # 测试集中的问题
                x_train_2 = []  # 测试集中的候选答案
                x_train_3 = []  # 测试集中的候选答案
        except:
            x_train_1.append(items[2])
            x_train_2.append(items[3])
            x_train_3.append(items[3])
            ques_list.append([x_train_1, x_train_2, x_train_3])
    fw = open('./td_data/train_question_complete.txt','w', encoding='utf-8')
    for ques in ques_list:
        question = ques[0][0]
        fw.write(question)
        fw.write('\n')
    return ques_list

def save_parse(in_file, out_file, lexer=True):
    fw = open(out_file, 'w', encoding='utf-8')
    for line in open(in_file, encoding='utf-8'):
        text = line.strip().replace('_', '')
        if lexer:
            lex_text = nlp.lexer(text)
            fw.write(json.dumps(lex_text))
            fw.write('\n')
        else:
            parser_text = nlp.depParser(text)
            fw.write(json.dumps(parser_text))
            fw.write('\n')


def save_parse2(in_file, out_file, lexer = True):
    qa_text = pd.read_csv(in_file, sep=' ', encoding='utf-8', names=['label', 'q_id', 'ques', 'ans', 'ans_id'])
    qa_text['ques'] = qa_text['ques'].apply(lambda x: ''.join(x.split('_')))
    qa_text['ans'] = qa_text['ans'].apply(lambda x: ''.join(x.split('_')))
    if lexer:
        # qa_text['ques'] = qa_text['ques'].apply(lambda x: nlp.lexer(x))
        qa_text['ans'] = qa_text['ans'].apply(lambda x: nlp.lexer(x))
    else:
        # qa_text['ques'] = qa_text['ques'].apply(lambda x: nlp.depParser(x))
        qa_text['ans'] = qa_text['ans'].apply(lambda x: nlp.depParser(x))
    qa_text.to_csv(out_file, sep=' ', encoding='utf-8')
    print("句子解析结束！")
    return
if __name__ == '__main__':
    # text = "你看的是上半部吧,下半部通常就说《赤壁下》啊。但有个名字叫《决战天下》的,反而没什么人提。"
    nlp = NLP()
    # lex_text = nlp.lexer(text)
    # paser_text = nlp.depParser(text)
    # print(len(lex_text['items']))
    # print(paser_text['items'])
    ######==========词法分析后的结果========######
    # save_parse('./td_data/train_question.txt', './td_data/lexer_ques.txt', lexer=True)      #训练集中的问题
    # save_parse('./td_data/test_question.txt', './td_data/lexer_ques_test.txt', lexer=True)    #测试集中的问题

    # save_parse2('./td_data/test_data.txt', './td_data/parse_test_data.txt', lexer=False)

    ######==========词法分析后的结果========######

    ######==========句法依存分析后的结果========######
    # save_parse('./td_data/train_question.txt', './td_data/parser_ques.txt', lexer=False)      #训练集中的问题
    # save_parse('./td_data/test_question.txt', './td_data/parser_ques_test.txt', lexer=False)    #测试集中的问题
    ######==========句法依存分析后的结果========######

    # train_ques_list = get_ques()
    # test_ques_list = get_ques()


    ###########============数据转换============############
    # trans_data(train_data_file, './td_data/train_data_complete.txt')
    # trans_data(test_data_file, './td_data/test_data.txt')
    ####=======数据转换======####

    # read_raw('./td_data/train_data.txt')


    ###########=======文档的padding操作=======##########
    # max_len = get_max_len()
    # padding('./td_data/train_data_complete.txt', './td_data/pd_train_data_complete.txt', max_len)
    # padding('./td_data/test_data.txt', './td_data/test_data2.txt', max_len)
    # padding('./td_data/rw_train_data.txt', './td_data/rw_train_data2.txt', max_len)
    ###########=======文档的padding操作=======##########


    # load_vectors()
    # build_vocab()
    # right_answer()
    # testList =load_test_and_vectors()
    # vocab = build_vocab()
    # load_vocabs()
    # load_data_test(testList, vocab)
    111