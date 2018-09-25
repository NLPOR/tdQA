from qdl_model import TextCNN
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
import json,logging
from qdl_helper import *
logging.getLogger().setLevel(logging.INFO)
def train_cnn(shuffled = True):
	train_file = './td_data/train_data_complete.txt'
	train_data = pd.read_csv(train_file, sep=' ', names=['label', 'q_id', 'question', 'anwser', 'anwser_id'])
	q_raw = train_data['question'].apply(lambda x: x.replace('_', ' ') if isinstance(x, str) else str(x)).tolist()
	a_raw = train_data['anwser'].apply(lambda x: x.replace('_', ' ') if isinstance(x, str) else str(x)).tolist()
	y_raw = train_data['label']								#得到所有的语料集

	parameter_file = './config/parameters.json'
	params = json.loads(open(parameter_file).read())    #读取参数文件

	"""Step 1: pad each sentence to the same length and map each word to an id"""
	"""第一步：将句子长度按照统一长度进行padding"""
	# max_document_length = max([len(x.split(' ')) for x in x_raw])
	max_len_a = 400
	max_len_q = 64

	logging.info('The maximum length of all anwser sentences: {}'.format(max_len_a))
	vocab_processor_a = learn.preprocessing.VocabularyProcessor(max_len_a)
	vocab_processor_a.fit(a_raw)


	logging.info('The maximum length of all question sentences: {}'.format(max_len_q))
	vocab_processor_q = learn.preprocessing.VocabularyProcessor(max_len_q)
	vocab_processor_q.fit(q_raw)


	labels = [0,1]
	one_hot = np.zeros((len(labels), len(labels)), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))
	y_raw = y_raw.apply(lambda y: label_dict[y]).tolist()
	y_raw = np.array(y_raw)
	"""Step 2: split the original dataset into train and test sets"""
	x_train_q, x_dev_q, y_train, y_dev = train_test_split(q_raw, y_raw, test_size=0.1, random_state=42)
	x_train_a, x_dev_a, y_train, y_dev = train_test_split(a_raw, y_raw, test_size=0.1, random_state=42)

	# """Step 3: shuffle the train set and split the train set into train and dev sets"""
	# if shuffled:
	# 	shuffle_indices = np.random.permutation(np.arange(len(y_train)))
	# 	x_train_q = x_train_q[shuffle_indices]
	# 	x_train_a = x_train_a[shuffle_indices]
	# 	y_train = y_train[shuffle_indices]

	logging.info('x_train: {}, x_dev: {}'.format(len(x_train_q), len(x_dev_q)))


	"""Step 5: build a graph and cnn object"""
	graph = tf.Graph()
	with graph.as_default():
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			cnn = TextCNN(
				q_length = 64,
				a_length= 400,
				num_classes = 2,
				vocab_size = len(vocab_processor_a.vocabulary_),
				embedding_size = params['embedding_dim'],
				filter_sizes = list(map(int, params['filter_sizes'].split(","))),
				num_filters = params['num_filters'],
				l2_reg_lambda = params['l2_reg_lambda']
            )
			global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.AdamOptimizer(1e-3)
			grads_and_vars = optimizer.compute_gradients(cnn.loss)
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

			timestamp = str(int(time.time()))
			out_dir = os.path.abspath(os.path.join('models', os.path.curdir, "trained_model_" + timestamp))

			checkpoint_dir = os.path.abspath(os.path.join('models', out_dir, "checkpoints"))
			checkpoint_prefix = os.path.join(checkpoint_dir, "model")
			if not os.path.exists(checkpoint_dir):
				os.makedirs(checkpoint_dir)
			saver = tf.train.Saver(tf.global_variables())

			# One training step: train the model with one batch
			def train_step(x_batch, x_batch2, y_batch):
				#对每一个x_batch进行训练，训练的dropout为0.5的神经元有效
				feed_dict = {
					cnn.input_x: x_batch,
					cnn.input_x2: x_batch2,
					cnn.input_y: y_batch,
					cnn.dropout_keep_prob: params['dropout_keep_prob']}
				_, step, loss, acc, prediction = sess.run([train_op, global_step, cnn.loss, cnn.accuracy,cnn.predictions], feed_dict)

			# One evaluation step: evaluate the model with one batch
			def dev_step(x_batch, x_batch2, y_batch):
				#对每一个batch进行评估，测试预测时全部神经元都有效
				feed_dict = {
					cnn.input_x: x_batch,
					cnn.input_x2: x_batch2,
					cnn.input_y: y_batch,
					cnn.dropout_keep_prob: 1.0}
				step, loss, acc, num_correct,prediction = sess.run([global_step, cnn.loss, cnn.accuracy, cnn.num_correct,cnn.predictions], feed_dict)
				return num_correct,prediction

			# Save the word_to_id map since predict.py needs it
			vocab_processor_q.save(os.path.join(out_dir, "vocab_q.pickle"))
			vocab_processor_a.save(os.path.join(out_dir, "vocab_a.pickle"))
			sess.run(tf.global_variables_initializer())

			# Training starts here，训练过程从这里开始
			train_batches = batch_iter(list(zip(x_train_q, x_train_a, y_train)), params['batch_size'], params['num_epochs'])#得到训练的batch数据
			best_accuracy, best_at_step = 0, 0

			"""Step 6: train the cnn model with x_train and y_train (batch by batch)"""
			for train_batch in train_batches:
				#对每一个batch进行训练
				x_q_batch, x_a_batch, y_train_batch = zip(*train_batch)
				x_q_batch = np.array(list(vocab_processor_q.transform(x_q_batch)))
				x_a_batch = np.array(list(vocab_processor_a.transform(x_a_batch)))

				train_step(x_q_batch, x_a_batch, y_train_batch)
				current_step = tf.train.global_step(sess, global_step)

				"""Step 6.1: evaluate the model with x_dev and y_dev (batch by batch)"""
				if current_step % params['evaluate_every'] == 0:#20步评估一次
					dev_batches = batch_iter(list(zip(x_dev_q, x_dev_a, y_dev)), params['batch_size'], 1)
					total_dev_correct = 0
					dev_predictions = []
					for dev_batch in dev_batches:
						x_dev_batch, x_dev_batch2, y_dev_batch = zip(*dev_batch)
						x_dev_batch = np.array(list(vocab_processor_q.transform(x_dev_batch)))
						x_dev_batch2 = np.array(list(vocab_processor_a.transform(x_dev_batch2)))
						num_dev_correct, dev_prediction = dev_step(x_dev_batch, x_dev_batch2, y_dev_batch)
						total_dev_correct += num_dev_correct
						dev_predictions = np.concatenate([dev_predictions, dev_prediction])
					print("验证集上的最后预测结果：")
					print(dev_predictions)
					dev_accuracy = float(total_dev_correct) / len(y_dev)
					logging.critical('Accuracy on dev set: {}'.format(dev_accuracy))

					"""Step 6.2: save the model if it is the best based on accuracy on dev set"""
					if dev_accuracy >= best_accuracy:
						best_accuracy, best_at_step = dev_accuracy, current_step
						path = saver.save(sess, checkpoint_prefix, global_step=current_step)
						logging.critical('Saved model at {} at step {}'.format(path, best_at_step))
						logging.critical('Best accuracy is {} at step {}'.format(best_accuracy, best_at_step))
train_cnn()

