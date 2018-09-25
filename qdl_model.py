import numpy as np
import tensorflow as tf
class TextCNN(object):
	def __init__(self, q_length, a_length, num_classes, vocab_size, embedding_size,
				 filter_sizes, num_filters, l2_reg_lambda=0.0):
		'''
		:param sequence_length: 表示文本长度，多少个词
		:param num_classes: 	待分类的类别个数
		:param vocab_size: 		词库的大小，表示构建的词库有多大
		:param embedding_size:  词向量维度大小
		:param filter_sizes:	卷积核的尺寸，是一个列表的形式[3,4,5]
		:param num_filters:		卷积核的个数
		:param l2_reg_lambda:	正则化系数
		'''
		with tf.name_scope('input'):	# 一个输入的命名空间
			self.input_x = tf.placeholder(tf.int32, [None, q_length], name='input_x')
			#输入问题

			self.input_x2 = tf.placeholder(tf.int32, [None, a_length], name='input_x2')
			#输入答案

			self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
			# 这个placeholder的数据输入类型为float，（样本数*类别）的tensor

		with tf.name_scope('dropout'):
			self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
			# placeholder表示图的一个操作或者节点，用来喂数据，进行name命名方便可视化

		# Keeping track of l2 regularization loss (optional)
		l2_loss = tf.constant(0.0)

		# Embedding layer
		with tf.device('/cpu:0'), tf.name_scope('embedding'):
			W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W')

			# tf.summary.histogram('embedding',W)								#这个是tensorboard画图的

			self.embedded_chars_q = tf.nn.embedding_lookup(W, self.input_x)		# 也就是最后问题的输入词向量

			self.embedded_chars_a = tf.nn.embedding_lookup(W, self.input_x2)	# 也就是最后答案的输入词向量


			#operation2，input_x的tensor维度为[none,seq_len],那么这个操作的输出为non*seq_len*em_size

			self.embedded_chars_expanded_q = tf.expand_dims(self.embedded_chars_q, -1)
			self.embedded_chars_expanded_a = tf.expand_dims(self.embedded_chars_a, -1)
			#增加一个维度，变成batch_size*seq_len*em_size*channel(=1)的4维tensor,符合图像的习惯

		# Create a convolution + maxpool layer for each filter size
		pooled_outputs_q = [] #空的list
		pooled_outputs_a = []
		for i, filter_size in enumerate(filter_sizes):#比如（0,3），（1,4），（2,5）
			with tf.name_scope('conv-maxpool-%s' % filter_size):				    # 循环一次建立一个名称为如“conv-ma-3”的模块
				# Convolution Layer
				filter_shape = [filter_size, embedding_size, 1, num_filters]	    #卷积核的参数，[高，宽，通道数，卷积核个数]

				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')	#卷积核的初始化

				# tf.summary.histogram('convW-%s' % filter_size, W)				    #tensorboard画图

				b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')    # 偏置b，维度为卷积核个数的tensor

				# tf.summary.histogram('convb-%s' % filter_size,b)				    #tensorboard画图

				conv = tf.nn.conv2d(				#卷积运算
					self.embedded_chars_expanded_q,	#输入特征矩阵
					W,								#初始化的卷积核矩阵
					strides=[1, 1, 1, 1],			#划窗移动距离[1, 横向距离， 纵向距离, 1]
					padding='VALID',				#边缘是否补0
					name='conv')


				h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')	#卷积之后使用relu（）激活函数去线性化


				# Maxpooling over the outputs
				pooled = tf.nn.max_pool(			#池化运算
					h,								#卷积后的输入矩阵
					ksize=[1, q_length - filter_size + 1, 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name='pool')
				pooled_outputs_q.append(pooled)
			# 每个卷积核和pool处理一个样本后得到一个值，这里维度如batchsize*1*1*卷积核个数三种卷积核，appen3次
				conv = tf.nn.conv2d(  # 卷积运算
					self.embedded_chars_expanded_a,  # 输入特征矩阵
					W,  # 初始化的卷积核矩阵
					strides=[1, 1, 1, 1],  # 划窗移动距离[1, 横向距离， 纵向距离, 1]
					padding='VALID',  # 边缘是否补0
					name='conv')

				h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')  # 卷积之后使用relu（）激活函数去线性化

				# Maxpooling over the outputs
				pooled = tf.nn.max_pool(  # 池化运算
					h,  # 卷积后的输入矩阵
					ksize=[1, a_length - filter_size + 1, 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name='pool')
				pooled_outputs_a.append(pooled)

		# Combine all the pooled features
		num_filters_total = num_filters * len(filter_sizes)					#每种卷积核个数与卷积种类的积

		self.h_pool_a = tf.concat(pooled_outputs_a,3)	# 将outputs在第4个维度上拼接，如本来是128*1*1*64的结果3个，拼接后为128*1*1*192的tensor
		self.h_pool_q = tf.concat(pooled_outputs_q,3)

		self.h_pool_flat_q = tf.reshape(self.h_pool_q, [-1, num_filters_total]) #  将最后结果reshape为128*192的tensor
		self.h_pool_flat_a = tf.reshape(self.h_pool_a, [-1, num_filters_total])

		self.h_pool_flat = tf.concat([self.h_pool_flat_q, self.h_pool_flat_a],1)


		# Add dropout
		with tf.name_scope('dropout'):				# 添加一个"dropout"的模块，里面一个操作，输出为dropout过后的128*192的tensor
			self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)		#使用dropout机制防止过拟合


		# Final (unnormalized) scores and predictions
		with tf.name_scope('output'):	#全连接操作，到输出层，注意这里用的是get_variables
			W = tf.get_variable(
				'W',
				shape=[num_filters_total*2, num_classes],		#	num_filters_total个features分num_classes类
				initializer=tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')		#输出层的偏置

			l2_loss += tf.nn.l2_loss(W)				#对全连接层的W使用l2_loss正则
			l2_loss += tf.nn.l2_loss(b)				#对全连接层的b使用l2_loss正则

			self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')			# 相当于tf.nn.matmul(self.h_drop, W) + b
			self.predictions = tf.argmax(self.scores, 1, name='predictions')		# 转换成one-hot的编码形式


		# Calculate mean cross-entropy loss
		with tf.name_scope('loss'):#定义一个”loss“的模块
			losses = tf.nn.softmax_cross_entropy_with_logits(labels = self.input_y, logits = self.scores) #  交叉熵损失函数
			self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss	#计算loss（包含正则化系数）

			# tf.summary.scalar('loss',self.loss)	#tensorboard 画图形式

		# Accuracy
		with tf.name_scope('accuracy'):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
			# operation2，计算均值即为准确率，名称”accuracy“


		with tf.name_scope('num_correct'):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.num_correct = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct')

