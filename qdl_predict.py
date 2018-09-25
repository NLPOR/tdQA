import os
import sys
import json
import logging
from qdl_helper import *
# from data_helper import clean_str
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn.metrics import classification_report
logging.getLogger().setLevel(logging.INFO)

def get_submission(path,predictions):
	test_data = pd.read_csv('./td_data/test_data.txt', sep=' ', names=['label', 'q_id', 'question', 'anwser', 'anwser_id'])
	test_id = test_data['anwser_id']
	predictions = pd.DataFrame({'labels':predictions})
	# label_map = {0.0:"amber", 1.0:"crisis", 2.0:"green", 3.0:"red"}
	# predictions = predictions['labels'].map(label_map)
	result = pd.concat([test_id, pd.Series(predictions)], axis=1)
	result.to_csv(path, sep=',', index=None, header=None)
	return predictions


def predict_unseen_data():
	"""Step 0: load trained model and parameters"""
	params = json.loads(open('./config/parameters.json').read())
	# checkpoint_dir = sys.argv[1]
	checkpoint_dir = './models/trained_model_1522729239/'
	if not checkpoint_dir.endswith('/'):
		checkpoint_dir += '/'
	checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir + 'checkpoints')#加载最近保存的模型
	logging.critical('Loaded the trained model: {}'.format(checkpoint_file))

	"""Step 1: load data for prediction"""
	# test_file = sys.argv[2]
	test_data = pd.read_csv('./td_data/test_data.txt', sep=' ', names=['label', 'q_id', 'question', 'anwser', 'anwser_id'])
	test_id = test_data['anwser_id']
	# labels.json was saved during training, and it has to be loaded during prediction
	labels = [0,1]
	one_hot = np.zeros((len(labels), len(labels)), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))

	#得到x_test
	q_raw = test_data['question'].apply(lambda x: x.replace('_', ' ') if isinstance(x, str) else str(x)).tolist()
	a_raw = test_data['anwser'].apply(lambda x: x.replace('_', ' ') if isinstance(x, str) else str(x)).tolist()
	y_raw = pd.read_csv('./td_data/submit_sample.txt', sep=',', names=['id', 'label'])['label']			#得到所有的语料集
	y_raw = y_raw.apply(lambda y: label_dict[y]).tolist()
	y_raw = np.array(y_raw)

	logging.info('The number of x_test: {}'.format(len(y_raw)))

	#得到y_test

	vocab_path_q = os.path.join(checkpoint_dir, "vocab_q.pickle")
	vocab_path_a = os.path.join(checkpoint_dir, "vocab_a.pickle")
	vocab_processor_q = learn.preprocessing.VocabularyProcessor.restore(vocab_path_q)
	vocab_processor_a = learn.preprocessing.VocabularyProcessor.restore(vocab_path_a)
	q_raw = np.array(list(vocab_processor_q.transform(q_raw)))
	a_raw = np.array(list(vocab_processor_a.transform(a_raw)))
	print(q_raw.shape)
	print(a_raw.shape)
	"""Step 2: compute the predictions"""
	graph = tf.Graph()
	with graph.as_default():
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		sess = tf.Session(config=session_conf)

		with sess.as_default():
			saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
			saver.restore(sess, checkpoint_file)

			input_x = graph.get_operation_by_name("input/input_x").outputs[0]
			input_x2 = graph.get_operation_by_name("input/input_x2").outputs[0]
			# dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
			dropout_keep_prob = graph.get_operation_by_name("dropout/dropout_keep_prob").outputs[0]
			predictions = graph.get_operation_by_name("output/predictions").outputs[0]
			batches = batch_iter(list(zip(q_raw, a_raw)), params['batch_size'], 1, shuffle=False)
			all_predictions = []
			for batch in batches:
				q_batch, a_batch = zip(*batch)
				batch_predictions = sess.run(predictions, {input_x: q_batch, input_x2:a_batch, dropout_keep_prob: 1.0})
				all_predictions = np.concatenate([all_predictions, batch_predictions])


	if y_raw is not None:
		y_test = np.argmax(y_raw, axis=1)
		print((all_predictions).tolist())
		correct_predictions = sum(all_predictions == y_test)
		print(classification_report(y_test, all_predictions, digits=3))
		logging.critical('The accuracy is: {}'.format(correct_predictions / float(len(y_test))))
		# get_submission('sub1.tsv',all_predictions)

		logging.critical('The prediction is complete')

if __name__ == '__main__':
	# python predict.py ./trained_model_1478649295/ ./data/small_samples.json
	predict_unseen_data()
