# -*- coding: utf-8 -*-
import tensorflow as tf
import os, json
import time
import td_data_helper
from tdqa_cnn import InsQACNN
import numpy as np
import datetime
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 500, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 1, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1, "Save model after this many steps (default: 100)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# =================Data Preparatopn================= #
# ================================================== #

# Load data
print("Loading data...")

vocab = td_data_helper.load_vocabs()             #所有训练集的词表集合
alist = td_data_helper.read_alist()              #训练集中所有的所有候选答案

raw = td_data_helper.read_raw()                  #数据的所有结果，形式为列表：[[1, qud:0, question, answer],...,[]]

x_train_1, x_train_2, x_train_3 = td_data_helper.load_data_6(vocab, alist, raw, FLAGS.batch_size)

testList = td_data_helper.load_test_and_vectors()

print("Load done...")

result = './results/sub_test'

with tf.Graph().as_default():
  with tf.device("/gpu:0"):
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = InsQACNN(
            sequence_length=x_train_1.shape[1],
            batch_size=FLAGS.batch_size,
            vocab_size=len(vocab),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-1)
        #optimizer = tf.train.GradientDescentOptimizer(1e-2)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        saver = tf.train.Saver(tf.global_variables())

        #### Initialize all variables
        # sess.run(tf.global_variables_initializer())

        saver.restore(sess, "myNet/tdQA_CNN_model_5000.ckpt")


        def dev_step():
          scoreList = []
          time_stamp = str(int(time.time()))
          #每个问题的候选答案cos得分
          of = open(result + '_' + time_stamp, 'w')

          fw_result_score = open('./results/qa_scores.txt', 'w')

          test_ques_list = td_data_helper.load_data_test(testList, vocab)   #测试集的问题列表
          for test_ques in range(0,len(test_ques_list)):

              print('test_ques:',test_ques)

              x_test_1, x_test_2, x_test_3 = test_ques_list[test_ques][0], test_ques_list[test_ques][1], test_ques_list[test_ques][2]

              feed_dict = {
                  cnn.input_x_1: x_test_1,
                  cnn.input_x_2: x_test_2,
                  cnn.input_x_3: x_test_3,
                  cnn.dropout_keep_prob: 1.0
              }
              batch_scores = sess.run([cnn.cos_12], feed_dict)         #每个问题的候选答案对应的得分
              # print('acc:', acc)
              fw_result_score.write(json.dumps(batch_scores[0].tolist()))
              fw_result_score.write('\n')


              for i in range(len(batch_scores[0])):
                  of.write(json.dumps(batch_scores[0].tolist()[i]))
                  of.write('\n')
              of.write('======下一个问题=====')
              of.write('\n')




        print("\nStart Testing:")
        dev_step()
        print("")

        temp_result = './results/qa_scores.txt'  # 测试集的问题和候选答案得分情况


        def get_qa_id(in_file='./td_data/test_data_sample.json'):
            string = open(in_file, 'r').read()
            s_time = datetime.datetime.now()
            print('开始时间：', s_time)
            data = json.loads(string)
            answer_id = []
            for question in data:
                ques_id = question['item_id']  # 问题id
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
            for line in open(result, 'r'):
                score = json.loads(line)  # 得分列表
                answer_list = [0] * len(score)
                print('候选答案得分：')
                print(score)
                score_rank = np.array(score).argsort()
                print("候选答案可能情况的从小到大的排序：")
                print(score_rank)
                for i in range(len(score_rank)):
                    if i < len(score_rank) - 3:
                        answer_list[score_rank[i]] = 0
                    else:
                        answer_list[score_rank[i]] = 1
                print("该问题的答案情况：")
                print(answer_list)
                id_anwsers.append(answer_list)
            return id_anwsers


        def make_submission(sub_file):
            fw = open(sub_file, 'w')
            qa_id = get_qa_id()  # 答案id
            id_anwsers = rank_answer(temp_result, 3)
            anwsers = []  # 是否为答案（0或者1）

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
            test = pd.read_csv(test_file, header=None, sep=',', names=['id', 'answer'])
            sub = pd.read_csv(sub_file, header=None, sep=',', names=['id', 'answer'])
            test_answer = test['answer']
            test_sub = sub['answer']
            print("准确率：")
            print(accuracy_score(test_answer, test_sub))
            print("正确答案和错误答案的prf值：")
            print(classification_report(test_answer, test_sub))


        if __name__ == '__main__':
            make_submission('./results/submission/sub.txt')
            print_acc()
            print('testing done.')

