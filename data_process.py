import pandas as pd
import numpy as np
import time
from gensim.models.word2vec import Word2Vec



def clear_text(text):
    return text

def reduce_sentence(sentences, n=6):
    '''
    截取句子
    :param sentences:
    :param n:
    :return:
    '''
    if len(sentences) > n:                      #   如果大于4个句子
        return sentences[:3] + sentences[-3:]   #   返回前2个和后两个
    else:                                       #   小于等于4个
        return sentences                        #   返回所以句子




def get_sentences(text):
    def cut_sentences(sentence):
        if not isinstance(sentence, str):
            sentence = str(sentence)
        puns = ['。', '!', '?']
        tmp = []
        i = 0
        for ch in sentence:
            tmp.append(ch)
            if ch in puns:
                i += 1
                print('第{}个句子。'.format(i))
                yield ''.join(tmp)
                tmp = []
        yield ''.join(tmp)
    sentences = []
    for i in cut_sentences(text):
        sentences.append(i)
    print(sentences)
    return sentences

def get_word2vec(corpus, outp1, outp2):
    t1 = time.time()
    print("开始训练时间：", t1)
    model = Word2Vec(corpus.tolist(), size=128, window=5, min_count=5,
                     workers=5)

    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)
    t2 = time.time()
    print("结束始训练时间：", t2)
    print(model.most_similar(u"高速"))
    return
train_data = pd.read_csv('./td_data/train_data_complete.txt',
                              encoding='utf-8',
                              names=['label', 'q_id', 'question', 'anwser', 'anwser_id'],
                              sep=' ')
test_data = pd.read_csv('./td_data/test_data.txt',
                              encoding='utf-8',
                              names=['label', 'q_id', 'question', 'anwser', 'anwser_id'],
                              sep=' ')

corpus1 = train_data['question'] + '_' + train_data['anwser']
corpus2 = test_data['question'] + '_' +test_data['anwser']

corpus = pd.concat([corpus1.apply(lambda x:x.split('_') if isinstance(x, str) else str(x)),
                    corpus2.apply(lambda x:x.split('_') if isinstance(x, str) else str(x))], ignore_index=True)

train_data['anwser_sent'] = train_data['anwser'].apply(get_sentences)

train_data['cut_sent'] = train_data['anwser_sent'].apply(reduce_sentence)

text = train_data['question'] + '_' + train_data['cut_sent'].apply(lambda x: '_'.join(x))
text.to_csv('qa_data.txt',index=None, encoding='utf-8')


print()
################=====================################

# get_word2vec(corpus, 'word2vectModel_200/qa_corpus_128d.model', 'word2vectModel_200/qa_corpus_128d.vec')