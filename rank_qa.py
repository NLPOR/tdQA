import json
import numpy as np
import datetime
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from clfl_question import main

result = './results/qa_scores8000'          #测试集的问题和候选答案得分情况

def get_qa_id(in_file = './td_data/test_data_sample.json'):
    string = open(in_file, 'r', encoding='utf-8').read()
    s_time = datetime.datetime.now()
    print('开始时间：', s_time)
    data = json.loads(string)
    answer_id = []
    for question in data:
        ques_id = question['item_id']      # 问题id
        all_answer = question['passages']  # 问题的所有候选答案
        for message in all_answer:
            answer_id.append(message['passage_id'])
    print(answer_id)
    print(len(answer_id))
    return answer_id

def rank_answer(result, n):
    '''
    :param result:  问题的结果得分列表
    :param n:       按照得分top-n作为正确答案，也即为1
    :return:        最后的问题答案候选列表 [ [q1_a,....], [q2_a,....],....,[]   ]
    '''
    id_anwsers = []
    for line in open(result,'r', encoding='utf-8'):
        score = json.loads(line)               #得分列表
        answer_list = [0] * len(score)
        print('候选答案得分：')
        print(score)
        score_rank = np.array(score).argsort()
        print("候选答案可能情况的从小到大的排序：")
        print(score_rank)
        for i in range(len(score_rank)):
            if i < len(score_rank)-n:
                answer_list[score_rank[i]] = 0
            else:
                answer_list[score_rank[i]] = 1
        print("该问题的答案情况：")
        print(answer_list)
        id_anwsers.append(answer_list)
    return id_anwsers

def make_submission(sub_file, flag =1):
    if flag:
        fw = open(sub_file, 'w', encoding='utf-8')
        qa_id = get_qa_id()                         #答案id
        id_anwsers = rank_answer(result,6)
        anwsers = []                                #是否为答案（0或者1）

        for x in id_anwsers:
            anwsers.extend(x)
        for id, anwser in zip(qa_id, anwsers):
            fw.write(str(id))
            fw.write(',')
            fw.write(str(anwser))
            fw.write('\n')
        fw.close()
    else:
        fw = open(sub_file, 'w', encoding='utf-8')
        qa_id = get_qa_id()                         # 答案id
        id_anwsers = rank_answer2(qa_scores_file = './results/qa_scores8000', n=6)
        anwsers = []                                # 是否为答案（0或者1）

        for x in id_anwsers:
            anwsers.extend(x)
        for id, anwser in zip(qa_id, anwsers):
            fw.write(str(id))
            fw.write(',')
            fw.write(str(anwser))
            fw.write('\n')
        fw.close()

def print_acc():
    test_file = './td_data/submit_sample.txt'
    sub_file = './results/submission/sub.txt'
    test = pd.read_csv(test_file, header=None, sep= ',',names=['id','answer'])
    sub = pd.read_csv(sub_file, header=None, sep= ',',names=['id','answer'])
    test_answer = test['answer']
    test_sub = sub['answer']
    print("准确率：")
    print(accuracy_score(test_answer, test_sub))
    print("正确答案和错误答案的prf值：")
    print(classification_report(test_answer, test_sub))


def rank_answer2(qa_scores_file= './results/qa_scores8000' , n=6):
    """
    对问题进行分类后的二次排次
    :return:
    """
    qa_score = [np.array(json.loads(line.strip()), np.float32) for line in open(qa_scores_file, encoding='utf-8')]
    clf_ques = main()
    length = len(qa_score)
    i = 0
    for qa_score_list, clf_ques_list in zip(qa_score, clf_ques):
        index = np.where(clf_ques_list == 1.0)
        clf_ques[i][index] = qa_score[i][index]
        i+=1

    id_anwsers = []
    for score in clf_ques:
        answer_list = [0] * len(score)
        print('候选答案得分：')
        print(score)
        score_rank = np.array(score).argsort()
        print("候选答案可能情况的从小到大的排序：")
        print(score_rank)
        for i in range(len(score_rank)):
            if i < len(score_rank) - n:
                answer_list[score_rank[i]] = 0
            else:
                answer_list[score_rank[i]] = 1
        print("该问题的答案情况：")
        print(answer_list)
        id_anwsers.append(answer_list)
    return id_anwsers

if __name__ == '__main__':
    make_submission('./results/submission/sub.txt', flag=0)
    print_acc()
    # rank_answer2('./results/qa_scores100')
    1111