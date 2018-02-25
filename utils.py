import os
import pdb
from collections import Counter
import copy
from padding import pad_sequences
import torch 
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import numpy as np
from model import EncoderRnn, DecoderRnn
src = 'europarl-v8.fi-en.en'
trg = 'europarl-v8.fi-en.fi'
def load_data(path, lang):
	def compose(func1, func2):
		return lambda x:func1(func2(x))

	def read_raw(filename):
		with open(filename, 'r') as f:
			return f.readlines()
	def tokenize(lines):
		words = []
		for i,line in enumerate(lines[:5000]):
			lower_and_strip = compose(str.rstrip, str.lower)
			line = lower_and_strip(line)
			l = line.split()
			words.append(l)
		return words
	
	read_and_tokenize = compose(tokenize, read_raw)
	return read_and_tokenize(os.path.join(path, lang))
	
class Lang():
	def __init__(self, sequences, n_words):
		self.sequences = sequences
		self.n_words = n_words
		super(Lang, self).__init__()
	def compose(self, func1, func2):
		return lambda x:func1(func2(x))
	def tokenize(self, sequence):
		lower_and_strip = self.compose(str.rstrip, str.lower)
		sample = lower_and_strip(sequence)
		return sample.split()
	def build_vocab(self):
		self.all = []
		counter = Counter()
		def pool(tokens):
			self.all.extend(tokens)
		tokenize_and_pool = self.compose(pool, self.tokenize)
		for i in range(len(self.sequences)):
			tokenize_and_pool(self.sequences[i])
		return [word for word, _ in Counter(self.all).most_common(self.n_words - 3)]
		# return list(set(self.all))
	def word2index(self):
		index={}
		self.vocab = self.build_vocab()
		self.vocab.insert(0, 'zero'); self.vocab.extend(['UNK','$'])
		for word in self.vocab:
			index[word] = len(index)
		return index
	def index2word(self):
		self.index = self.word2index()
		return {k:i for i, k in self.index.items()}

def split_data(data):
	fr = 0.8
	split_index = int(fr*len(data))
	return split_index


