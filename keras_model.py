#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-09-17 17:03:53
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import os
from keras.engine import Input
from keras.layers import merge, Embedding, Dropout, Conv1D, Lambda, LSTM, Dense, concatenate, TimeDistributed
from keras import backend as K
from keras.models import Model


class  LanguageModel(object):
	"""docstring for  LanguageModel"""
	def __init__(self, config):
		# 输入
		self.questions = Input(shape= (config["questions_len"],),dtype = 'int32',name = "questions_base")
		self.good_answers = Input(shape = (config["answers_len"],), dtype = 'int32', name = "good_answers_base")
		self.bad_answers = Input(shape = (config["answers_len"],), dtype = 'int32', name = "bad_answers_base")

		self.config = config
		self._model = None # 模型
		self._scores = None # 得分
		self._answer = None
		self._qa_model = None

		self.training_model = None
		self.prediction_model = None

	def get_answer(self):
		if self._answer is None:
			self._answer = Input(shape = (self.config["answers_len"],), dtype='int32', name = "answer")
		return self._answer

	@abstractmethod
	def build(self):
		return

	def get_similarity(self):
		params = self.params
		# 计算相似度的方法
		similarity = params['mode']
		dot = lambda a,b: K.batch_dot(a,b,axes = 1)
		l2_norm = lambda a,b: K.sqrt(K.sum(K.square(a - b), axis = 1, keepdims = True))

		if similarity == 'cosine':
			return lambda x: dot(x[0],x[1]) / K.maximum(K.sqrt(dot(x[0],x[0]) * dot(x[1],x[1])),K.epsilon())
		elif similarity ==  'polynomial':
			return lambda x: (params['gamma'] * dot(x[0],x[1]) + params['c']) ** params['d']
		elif similarity == 	'sigmoid':
			return lambda x:K.tanh(params['gamma'] * dot(x[0],x[1]) + params['c'])
		elif similarity == 'rbf':
			return lambda x: K.exp(-1 * params['gamma'] * l2_norm(x[0],x[1])**2)
		elif similarity == 'eulidean':
			return lambda x: 1/ (1 + l2_norm(x[0],x[1]))
		elif similarity == 'exponential':
			return lambda x: K.exp(-1 * params['gamma'] * l2_norm(x[0],x[1]))
		elif similarity == 'gesd':
			eulidean = lambda x: 1 / (1 + l2_norm(x[0],x[1]))
			sigmoid = lambda x: 1 / (1 + K.exp(-1 * params['gamma'] * dot(x[0],x[1] + params['c'])))
			return lambda x: eulidean(x) * sigmoid(x)
		elif similarity == 'aesd':
			eulidean = lambda x: 0.5 / (1 + l2_norm(x[0],x[1]))
			sigmoid = lambda x: 0.5 / (1 + K.exp(-1 * params['gamma'] * dot(x[0],x[1] + params['c'])))
			return lambda x: eulidean(x) + sigmoid(x)
		else:
			raise Exception('Invalid similarity:{}'.format(similarity))
	# 定义输出层的相似度计算方法
	def get_QA_model(self):
		if self._model is None:
			self._model = self.build()
		if self._qa_model is None:
			# 
			question_output,answer_output = self._model
            # 防止过拟合
			dropout = Dropout(self.params.get('dropout'),0.2)
			# 相似度的计算方法
			similarity = self.get_similarity()
			qa_model = Lambda(similarity,output_shape = lambda _:(None,1))([dropout(question_output),
				                                                            dropout(answer_output)])
			self._qa_model = Model(inputs = [self.questions,self.get_answer(),outputs = qa_model,name = 'qa_model'])

	def complie(self,optimizer, **kwargs):
		qa_model = self.get_QA_model()

		good_similarity = qa_model([self.questions,self.good_answers])
		bad_similarity  =qa_model([self.questions,self.bad_answers])



		# 损失函数
		loss = Lambda(lambda x: K.relu(self.config['margin'] - x[0] + x[1]),
			          output_shape = lambda x:x[0])([good_similarity,bad_similarity])

		self.prediction_model = Model(inputs = [self.questions,self.good_answers],outputs = good_similarity,
			                          name = 'prediction_model')
		self.prediction_model.complie(loss = lambda y_true,y_pred: y_pred, optimizer = optimizer,**kwargs)

		self.training_model = Model(inputs = [self.questions,self.good_answers,self.bad_answers],outputs = loss,
			                        name = 'training_model')
		training_model.complie(loss = lambda y_true,y_pred: y_pred,optimizer = optimizer,**kwargs)

	def fit(self,x,**kwargs):
		assert self.training_model is not None, 'Must compile the model before fitting data'
		y = np.zeros(shape = (x[0].shape[0],))
		return self.training_model.fit(x,y,**kwargs)

	def predict(self,x):
		assert self.prediction_model is not None and isinstance(self.prediction_model,Model)
		return self.prediction_model.predict_on_batch(x)

	def save_weights(self,file_name,**kwargs):
		assert self.prediction_model is not None,'Must compile the model before saving weights'
		self.prediction_model.save_weights(file_name,**kwargs)
	def load_weights(self,file_name,**kwargs):
		assert self.prediction_model is not None,'Must compile the model loading weights'
		self.prediction_model.load_weights(file_name,**kwargs)

class EmbeddingModel(LanguageModel):
	"""docstring for EmbeddingModel"""
	def build(self):
		question = self.question
		answer = self.get_answer()

		weights = np.load(self.config['initial_embed_weights'])
		embedding = Embedding(input_dim = self.config['n_words'],
			                  output_dim = weights.shape[1],
			                  mask_zero = True,
			                  weights = [weights])
		question_embedding = embedding(question)
		answer_embedding = embedding(answer)


		maxpool = Lambda(lambda x: K.max(x,axis = 1, keepdims = False),output_shape = lambda x:(x[0],x[2]))
		maxpool.supports_masking = True
		question_pool = maxpool(question_embedding)
		answer_pool = maxpool(answer_embedding)

		return question_pool,answer_pool

class ConvolutionModel(LanguageModel):
	def build(self):
		assert self.config['questions_len'] == self.config['answers_len']

		question = self.questions
		answer = self.get_answer()

		weights = np.load(self.config['initial_embed_weights'])
		embedding = Embedding(input_dim = self.config['n_words'],
							  output_dim = weights.shape[1],
							  weights = [weights])
		question_embedding = embedding(question)
		answer_embedding = embedding(answer)
		# Dense()
		hidden_layer = TimeDistributed(Dense(200,activation = 'tanh'))
		# 输入层处理
		question_hl = hidden_layer(question_embedding)
		answer_hl = hidden_layer(answer_embedding)

		# 一维卷积核
		cnns = [Conv1D(kernel_size = kernel_size,
			            filters = 1000,
			            activation = 'tanh',
			            padding = 'same') for kernel_size in [2,3,5,7]]

		# 卷积层输出
		question_cnn = concatenate([cnn(question_hl) for cnn in cnns],axis = -1)
		answer_cnn = concatenate([cnn(answer_hl) for cnn in cnns],axis = -1)
		# 池化层输出
		maxpool = Lambda(lambda x: K.max(x,axis = 1,keepdims = False), output_shape = lambda x: (x[0],x[2]))
		maxpool.supports_masking = True
		# maxpooling层输出
		question_pool = maxpool(question_hl)
		answer_pool = maxpool(answer_hl)

		return question_pool, answer_pool
# 对输入数据进行三维的卷积，
class MCCNNs(LanguageModel):
	def build(self):
		question = self.questions
		answer = self.get_answer()

		weights = np.load(self.config['initial_embed_weights'])
		embedding = Embedding(input_dim = self.config['n_words'],
			                  output_dim = weights.shape[1],
			                  weights = [weights])
		question_embedding = embedding(question)
		answer_embedding = embedding(answer)

		hidden_layer = TimeDistributed(Dense(200,activation = 'tanh'))

		question_hl = hidden_layer(question_embedding)
		answer_hl = hidden_layer(answer_embedding)

		cnns = [Conv2D(filters = 3, 
			           kernel_size =(),
			           activation = 'tanh',
			           padding = 'same')]
		#question_cnn = 

		maxpool = Lambda( lambda x: K.max(x,axis = 1,keepdims = False),output_shape = lambda x: (x[0],x[2]))
		maxpool.supports_masking = True

		question_pool = maxpool(question_cnn)
		answer_pool = maxpool(answer_cnn)

		return question_pool,answer_pool

# LSTM -> cnn
class ConvolutionalLSTM(LanguageModel):
	"""docstring for ConvolutionalLSTM"""
	def build(self):
		question  = self.questions
		answer = self.get_answer()

		weights = np.load(self.config['initial_embed_weights'])
		embedding = Embedding(input_dim = self.config['n_words'],
							  output_dim = weights.shape[1],
							  weights = [weights])
		question_embedding = embedding(question)
		answer_embedding = embedding(answer)

		f_rnn = LSTM(141,return_sequences = True, implementation = 1)
		b_rnn = LSTM(141,return_sequences = True, implementation = 1, go_backwards = True)

		qf_rnn = f_rnn(question_embedding)
		qb_rnn = b_rnn(question_embedding)
		#
		question_pool = concatenate([qf_rnn,qb_rnn], axis = -1)

		af_rnn = f_rnn(answer_embedding)
		ab_rnn = f_rnn(answer_embedding)

		answer_pool = concatenate([af_rnn,ab_rnn], axis = -1)

		#cnn
		cnns = [Conv1D(kernel_size = kernel_size,
			           filters = 500,
			           activation = 'tanh',
			           padding = 'same') for kernel_size in [1,2,3,5]]

		question_cnn = concatenate([cnn(question_embedding) for cnn in cnns], axis = -1)
		answer_cnn = concatenate([cnn(answer_embedding) for cnn in cnns],axis = -1)

		maxpool = Lambda(lambda x: K.max(x,axis = 1, keepdims = False),output_shape = lambda x:(x[0],x[2]))
		maxpool.supports_masking = True
		question_pool = maxpool(question_cnn)
		answer_pool = maxpool(answer_cnn)

		return question_pool,answer_pool

class AttentionModel(object):
	"""docstring for AttentionModel"""
	def build(self):
		question = self.questions
		answer = self.get_answer()

		weights = np.load(self.config['initial_embed_weights'])
		embedding = Embedding(input_dim = self.config['n_words'],
			                  output_dim = weights.shape[1],
			                  weights = [weights])
		question_embedding = embedding(question)
		answer_embedding = embedding(answer)


		f_rnn = LSTM(141, return_sequences = True, consume_less = 'mem')
		b_rnn = LSTM(141, return_sequences = True,consume_less = 'mem', go_backwards = True)

		question_f_rnn = f_rnn(question_embedding)
		question_b_rnn = b_rnn(question_embedding)

		# 池化操作
		maxpool = Lambda( lambda x: K.max(x, axis = 1, keepdims = False),output_shape = lambda x:(x[0],x[2]))
		maxpool.supports_masking = True
		question_pool = merge([maxpool(question_f_rnn),maxpool(question_b_rnn)],mode = 'concat',concat_axis = -1)

		answer_f_rnn = f_rnn(answer_embedding)
		answer_b_rnn = b_rnn(answer_embedding)
		answer_pool = merge([maxpool(answer_f_rnn),maxpool(answer_b_rnn)],mode = 'concat',concat_axis = -1)

		return question_pool,answer_pool
