import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from padding import pad_sequences
import pdb
from utils import  Lang
from torch.utils.data.sampler import SubsetRandomSampler

class LanguageDataset(Dataset):
	def __init__(self, lang_file, root_dir, transform=None):
		with open(os.path.join(root_dir, lang_file), 'r') as f:
			self.sequences = f.readlines()[:50000]
		self.root_dir = root_dir
		self.transform = transform
	def __len__(self):
		return len(self.sequences)
	def __getitem__(self, idx):
		return self.transform(self.sequences[idx]) if self.transform else self.sequences[idx]

class Tokenize(object):
	def compose(self, func1, func2):
		return lambda x:func1(func2(x))
	def __call__(self, sample):
		lower_and_strip = self.compose(str.rstrip, str.lower)
		sample = lower_and_strip(sample)
		return sample.split()

class EncodeSeq(object):
	def __init__(self, vocab):
		self.vocab = vocab

	def __call__(self, sample):
		for i, token in enumerate(sample):
			# pdb.set_trace()
			if token in self.vocab:
				
				sample[i] = self.vocab[token]
			else:
				sample[i] = self.vocab['UNK']
		return sample

class Pad(object):
	def __init__(self, maxlen):
		self.maxlen = maxlen
	def __call__(self, sample):
		if len(sample) > self.maxlen:
			return sample[:self.maxlen]
		pad_required = (self.maxlen - len(sample))*[0]
		return sample + pad_required
class ToTensor(object):
	def __call__(self, sample):
		return torch.LongTensor(sample)


def train(train_x):
	for i_batch, sample_batched in enumerate(train_x):
		x = sample_batched
		print(x.size())

