#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-09-17 19:19:20
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import os
import json
import numpy as np

class Evaluator(object):
	"""docstring for Evaluator"""
	def __init__(self, conf,model,optimizer = None):
		if isinstance(conf,str):
			config = json.load(open(conf,'rb'))

		self.config = config
		self.model = model
		self.optimizer = optimizer
		self.data_path = config["data_path"]
		self.vocab = None
		self.reverse_vocab = None
		self.train_set = None
		self.test_set = None
		self._vocab = None
		self._reverse_vocab = None
		self._eval_sets = None  # 评估

	def load(self,name):
		return pickle.load(open(os.path.join(self.path.name),'rb'))

	def vocab(self):
		if self._vocab is None:
			self._vocab = self.load('vocabulary')
		return self._vocab
	def reverse_vocab(self):
		if self._reverse_vocab is None:
			vocab = self.vocab()
			self._reverse_vocab = dict((v.lower(),k) for k,v in vocab.items())
		return self._reverse_vocab

	def save_epoch(self.epoch):
		if not os.path.exists('models/'):
			os.makedirs('models/')
		self.model.save_weights('models/weights_epoch_%d.h5' % epoch,overwrite = True)

	def loag_epoch(self.epoch):
		asssert os.path.exists('models/weights_epoch_%d.h5' % epoch), 'Wegiths at epoch %d not found'%epoch
		self.model.load_weights('models/weights_epoch_%d.h5' % epoch)

	def convert(self,words):
		rvocab = self.reverse_vocab()
		if type(words) == str:
			words = words.strip().lower().split(' ')
		return [rvocab.get(w,0) for w in words]
	def revert(self,indices):
		vocab = self.vocab()
		return [vocab.get(i,'X') for i in indices]

	def padq(self,data):
		return self.pad(data,self.conf.get('question_len',None))
	def pada(self,data):
		return self.pad(data,self.conf.get('answer_len',None))

	def pad(self,data, len = None):
		from keras.preprocessing.sequence import pad_sequences
		return pad_sequences(data,maxlen = len,padding = 'post', truncating = 'post', value = 0)

	def get_times(self):
		return strftime('%Y-%m-%d %H:%M:%S', gmtime())

	def train(self):
		batch_size = self.params['batch_size']
		nb_epoch = self.params['nb_epoch']
		validation_split = self.params['validation_split']

		training_set = self.load('train')


		questions = list()
		good_answers = list()
		indices = list()

		for j,q in enumerate(training_set):
			questions += [q['question']] * len(q['answers'])

			good_answers += [self.answers[i] for i in q['answers']]
			indices += [j] * len(q['answers'])

		log('Began trainig at %s on %d samples' % (self.get_time(),len(questions)))

		questions = self.padq(questions)
		good_answers = self.pada(good_answers)
		val_loss = {'loss':1., 'epoch':0}

		for i in range(1,nb_epoch + 1):
			bad_answers = self.pada(random.sample(self.answers.values(), len(good_answers)))

			print('Fitting epoch %d' % i,file = sys.stderr)

			hist = self.model.fit([questions,good_answers,bad_answers],epoches=1,batch_size = batch_size,
				                  validation_split = validation_split,verbose = 1)
			if hist.history['val_loss'][0] < val_loss['loss']:
				val_loss = {'loss':hist.history['val_loss'][0], 'epoch': i }
			log('%s -- Epoch %d' % (self.get_time(),i) +
				'Loss = %.4f, Validation Loss = %.4f' % (hist.history['loss'][0],hist.history['val_loss'][0]) +
				'(Best: Loss = %.4f, Epoch = %d)' % (val_loss['loss'], val_loss['epoch']))

			self.save_epoch(i)
		return val_loss

	def prog_bar(self,so_far,total,n_bars = 20):
		n_complete = int(so_far * n_bars / total)
		if n_complete >= n_bars - 1:
			print('\r[' + '=' * n_bars + ']',end = '', file = sys.stderr)
		else:
			s = '\r[' + '=' * (n_complete - 1) + '>' + '.' * (n_bars - n_complete) + ']'
			print(s, end = '', file = sys.stderr)

	def eval_sets(self):
		if self._eval_sets is None:
			self._eval_sets = dict([(s, self.load(s)) for in ['dev','test1','test2']])

		return self._eval_sets

	def get_score(self, verbose = False):
		top1_ls = []
		mrr_ls = []
		for name, data in self.eval_sets().items():
			print('----- %s ------' % name)

			random.shuffle(data)

			if 'n_eval' in self.params:
				data = data[:self.params['n_eval']]

			c_1,c_2 = 0,0

			for i,d in enumerate(data):
				self.prog_bar(i,len(data))

				indices = d['good'] + d['bad']
				answers = self.pada([self.answers[i] for i in indices])
				questions = self.padq([d['question']] * len(indices))

				sim = self.model.predict([question,answers])

				n_good = len(d['good'])
				max_r = np.argmax(sims)
				max_n = argmax(sims[:n_good])

				r= rankdata(sims, method = 'max')

				if verbose:
					min_r = np.argmin(sims)
					amin_r = self.answers[indices[min_r]]
					amax_r = self.answers[indices[max_r]]
					amax_n = self.answers[indices[max_n]]

					print(' '.join(self,revert(d['question'])))
					print('Predicted:({})'.format(sim[max_r]) + ' '.join(self.revert(amax_n)))
					print('Worst:({})'.format(sims[min_r]) + ' '.join(self.revert(amin_r)))
				c_1 += 1 if max_r == max_n else 0
				c_2 += 1 / float(r[max_r] - r[max_n] + 1)
			top1 = c_1 / float(len(data))
			mrr = c_2 / float(len(data))

			del data

			print('Top-1 Precision: %f ' % top1)
			print('MRR: %f' % mrr)
			top1_ls.append(top1)
			mrr_ls.append(mrr)
		return top1_ls,mrr_ls

if __name__ == '__main__':
	
	# 配置文件
	conf = {
        'n_words': 22353, # 单词数
        'question_len': 150, # 问题长度
        'answer_len': 150, #答案长度
        'margin': 0.009, # 容差
        # 权重，词向量
        'initial_embed_weights': 'word2vec_100_dim.embeddings',
        # 训练参数
        'training': {
            'batch_size': 100,
            'nb_epoch': 2000,
            'validation_split': 0.1,
        },
        # 相似函数定义
        'similarity': {
            'mode': 'cosine',
            'gamma': 1,
            'c': 1,
            'd': 2,
            'dropout': 0.5,
        }
    }
    from keras_models import EmbeddingModel, ConvolutionModel, ConvolutionalLSTM
    evaluator = Evaluator(conf, model=ConvolutionModel, optimizer='adam')

    # train the model
    best_loss = evaluator.train()

    # evaluate mrr for a particular epoch
    # 取得最好的训练结果
    evaluator.load_epoch(best_loss['epoch'])
    top1, mrr = evaluator.get_score(verbose=False)
    log(' - Top-1 Precision:')
    log('   - %.3f on test 1' % top1[0])
    log('   - %.3f on test 2' % top1[1])
    log('   - %.3f on dev' % top1[2])
    log(' - MRR:')
    log('   - %.3f on test 1' % mrr[0])
    log('   - %.3f on test 2' % mrr[1])
    log('   - %.3f on dev' % mrr[2])

		