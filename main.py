#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-09-17 19:19:20
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import os
import json

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
	

		