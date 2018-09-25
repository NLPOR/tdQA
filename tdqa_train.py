import tensorflow as tf
import numpy as np
import os, json
import time
import datetime
import td_data_helper
from tdqa_cnn import InsQACNN
import operator

#print tf.__version__
#                       Parameters                     #
# ==================================================== #

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 500, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 3000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 200, "Save model after this many steps (default: 100)")

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

#  vectors = ''
print('x_train_1', np.shape(x_train_1))
print("Load done...")

###==================保存的结果文件===================###

result = './results/sub_test'

###==================保存的结果文件===================###



# ==================== Training ==================== #
# ================================================== #

with tf.Graph().as_default():
  with tf.device("/cpu:0"):
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

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph_def)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph_def)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch_1, x_batch_2, x_batch_3):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x_1: x_batch_1,
              cnn.input_x_2: x_batch_2,
              cnn.input_x_3: x_batch_3,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(num):
          scoreList = []
          time_stamp = str(int(time.time()))
          #每个问题的候选答案cos得分
          of = open(result + '_' + time_stamp, 'w', encoding='utf-8')

          fw_result_score = open('./results/qa_scores' + str(num), 'w', encoding='utf-8')

          test_ques_list = td_data_helper.load_data_test(testList, vocab)   #测试集的问题列表
          for test_ques in range(0,len(test_ques_list)):

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

          # of.write("测试集问题的得分列表：", scoreList)

          ################
          #
          # of = open(result, 'a', encoding='utf-8')
          #
          # of.write()


        # Generate batches
        # Training loop. For each batch..
        for i in range(FLAGS.num_epochs):
            x_batch_1, x_batch_2, x_batch_3 = td_data_helper.load_data_6(vocab, alist, raw, FLAGS.batch_size)
            train_step(x_batch_1, x_batch_2, x_batch_3)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(FLAGS.evaluate_every)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
