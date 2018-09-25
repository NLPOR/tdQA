import json
from collections import OrderedDict
import numpy as np
from collections import Counter
from dataHelper import get_ques
class GetFeature(object):
    """
    主要用来获取每个问题特征类
    """

    #1.表层特征提取


    #2.依存句法分析句法结构特征
    #  a.主干和疑问词及其附属成分的问题分类句法特征
    def get_main_parse(self):
        lexer_file = './td_data/lexer_ques.txt'
        parser_file = './td_data/parser_ques.txt'
        lexer_lines = open(lexer_file).readlines()
        parser_lines = open(parser_file).readlines()
        feature_lines = []
        for lexer_line, parse_line in zip(lexer_lines, parser_lines):
            feature_words = ['主语', '谓语', '宾语', '疑问词', '疑问词附属', '命名实体']  # 需要抽取出每个句子的成分
            ques_lexer = json.loads(lexer_line)   # 词法分析
            ques_parser = json.loads(parse_line)  # 句法分析

            text = ques_lexer['text']           # 原句内容
            lexer_info = ques_lexer['items']    # 句子所有词法信息
            parser_info = ques_parser['items']  # 句子的所有句法信息
            for lexer_item, parser_item in zip(lexer_info, parser_info):
                '''抽取问句中的相应特征：'''

                if parser_item['deprel'] in ['SBV']:                        #抽取主语
                    feature_words[0] = parser_item['deprel']

                if parser_item['postag'] == 'v':                            #抽取谓语
                    feature_words[1] = parser_item['deprel']

                if parser_item['deprel'] in ['FOB', 'DOB', 'VOB']:          #抽取宾语
                    feature_words[2] = parser_item['deprel']

                if parser_item['postag'] == 'r':
                    feature_words[3] = parser_item['word']  # 抽取疑问词
                    feature_words[4] = parser_info[parser_item['head'] - 1]['word']  # 疑问词附属词

                if lexer_item['ne'] != '':               #命名实体
                    feature_words[5] = lexer_item['item']
            feature_lines.append(feature_words)
        return feature_lines

    def get_vocab(self,texts):
        word_list = []
        for text in texts:
            for word in text.split('_'):
                if word in word_list:
                    continue
                else:
                    word_list.append(word)
        vocab_to_int = {c: i for i, c in enumerate(word_list)}
        int_to_vocab = dict(enumerate(word_list))
        vocab_to_zero = {c: 0 for  c in word_list}
        return vocab_to_int, int_to_vocab, OrderedDict(vocab_to_zero)

    def idf(self, word_list, texts, path = None):
        '''
        输入文本句子text，计算每个词的权重信息
        :param text:
        :return:
        '''
        idf_dict = OrderedDict()
        # 计算每个词的idf
        print("正在计算idf值")
        if path:
            with open(path, 'w') as f:
                f.write('word' + '\t' + 'idf' + '\n')
                for word in word_list:
                    count = 0
                    for text in texts:
                        if word in text.split('_'):
                            count += 1
                    idf_dict[word] = np.log(len(texts) / (count + 1))
                    f.write(word + '\t' + str(idf_dict[word]) + '\n')
                    print(word, idf_dict[word])
                print("保存完毕！一共{}个单词".format(len(word_list)))
        else:
            for word in word_list:
                count = 0
                for text in texts:
                    if word in text.split('_'):
                        count += 1
                idf_dict[word] = np.log(len(texts) / (count + 1))
                print(word, idf_dict[word])
        return idf_dict

    def tf_idf(self, text):
        '''
        输入问题集texts和一个问题text, 计算text的tfidf
        :param text:
        :return:
        '''
        train_ques_file = './td_data/train_question.txt'
        test_ques_file = './td_data/test_question.txt'

        train_lines = open(train_ques_file, 'r', encoding='utf-8').readlines()
        test_lines = open(test_ques_file, 'r', encoding='utf-8').readlines()

        train_lines = [line.strip() for line in train_lines]
        test_lines = [line.strip() for line in test_lines]
        lines = train_lines + test_lines

        vocab_to_int, int_to_vocab, vocab_to_zero= self.get_vocab(lines)
        word_list = [word for word in vocab_to_zero]

        idf_dict = self.idf(word_list, lines)       #得到texts中每个词的idf值


        text_word = text.split('_')
        text_word_count = Counter(text_word)        #每个词的词频字典
        for word in text_word:
            tf = text_word_count[word] / len(text_word)             #每个词的tf值
            idf = idf_dict[word]                    #每个词的idf值
            vocab_to_zero[word] = tf*idf
        return [vocab_to_zero[i] for i in vocab_to_zero]

    def syn_word_weights(self,parser_text):
        distance = []                       #每个句子每个词到根节点的距离
        paser_items = parser_text['items']
        for item in paser_items:
            d = 0
            if item['head'] == 0:
                distance.append(1.0)
            else:
                while(paser_items[item['head']-1]['head'] != 0):
                    d +=1                                                    #如果父节点是根节点距离+2
                    item['head'] = paser_items[item['head']-1]['head']       #该节点的
                d +=2
                distance.append(1.0/np.sqrt(d))
        return distance

    def semantics_qa(self):
        '''
        #3.问题句和答案句的深度语义相关特征学习
        :return:
        '''
        return

    def get_word2vec(self):
        return
from parser__lexer import *
def clf_ques():
    pass

def person(text):
    #判断一个答案是否属于人物类，是就返回1，不是就返回0
    if isinstance(text, list):
        for i in range(len(text)):
            try:
                if text[i]['ne'] == 'PER':
                #如果包含人物实体就暴力原答案记为1
                    return 1
            except:
                #如果词法分析出错了，也保留原答案，记为1
                return 1
    return 0

def digit(text):
    #判断一个答案是否包含数字类，是就返回1，不是就返回0
    #1.整数：只包含0到9这10个字符的数字
    #2.小数与分数包含点、斜线、百分号等
    #3.西方的写法整数部分每三位有一个逗号分隔符：
    #4.以汉字的形式：二十八万
    #5.混排形式，以汉字与数字结合的：5万3千
    #6.除了汉字中的数字之外，还包含了百分之、分之、点等字样 ：百分之三十九
    if isinstance(text, list):
        for i in range(len(text)):
            try:
                if text[i]['pos'] == 'm':
                    return 1
            except:
                return 1
    return 0

def location(text):
    #判断一个答案是否属于包含地点，是就返回1，不是就返回0
    if isinstance(text, list):
        for i in range(len(text)):
            try:
                if text[i]['ne'] == 'LOC':
                    return 1
            except:
                return 1
    return 0

def time(text):
    #判断一个答案是否包含时间，是就返回1，不是就返回0
    #判断一个答案是否属于包含地点，是就返回1，不是就返回0
    if isinstance(text, list):
        for i in range(len(text)):
            try:
                if text[i]['ne'] == 'TIME':
                    return 1
            except:
                return 1
    return 0

def filter_anwser(ques_anwser_list, question_label):
    '''
    判断问题的类型
    根据问题的类型过滤一部分答案
    主要工作是去过滤掉不可能是正确的候选答案
    :param ques_anwser_list: 问题答案的列表[ [[q1,q1,...q1], [a1,a2,,...,a]],... ]
    :param question_label: 对应问题的标签列表[0,1,2,3,....]与ques_anwser_list一一对应
    :return:
    '''
    result = []
    for q_a_list, q_label in zip(ques_anwser_list, question_label):

        q_result = np.ones(len(q_a_list[0]), dtype=np.float32)     #初始化答案,假设当前答案全部为正确答案
        if int(q_label) == 0:                 #人物类问题
            for i in range(len(q_a_list[0])):
                q_result[i] = person(q_a_list[1][i])

        elif int(q_label) == 1:               #地点类问题
            for i in range(len(q_a_list[0])):
                q_result[i] = location(q_a_list[1][i])

        elif int(q_label) == 2:               #数字类问题
            for i in range(len(q_a_list[0])):
                q_result[i] = digit(q_a_list[1][i])

        elif int(q_label) == 3:               #时间类问题
            for i in range(len(q_a_list[0])):
                q_result[i] = time(q_a_list[1][i])

        elif int(q_label) == 4:               #实体类问题
            pass

        elif int(q_label) == 5:               #描述类问题
            pass
        else:                                 #其他类型的问题
            pass
        result.append(q_result)
    # fw = open('./results/qa_lexer.txt', 'w', encoding='utf-8')
    # fw.write(json.dumps(result))
    return result

def main():
    # 问题答案列表
    ques_list = ques_list_lexer()

    # 问题标签列表
    question_label = [label.split('\t')[1].strip() for label in open('./td_data/pad_te_ques2.txt', encoding='utf-8')]

    results = filter_anwser(ques_list, question_label)


    return results


if __name__ == '__main__':

    main()
