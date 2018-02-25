import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

class EncoderRnn(nn.Module):
	def __init__(self, vocab_size, hidden_size, dropout, use_cuda):
		super(EncoderRnn, self).__init__()
		self.hidden_size = hidden_size
		self.use_cuda = use_cuda
		self.embedding = nn.Embedding(vocab_size, hidden_size)
		self.lstm = nn.LSTMCell(hidden_size, hidden_size)
		self.drop = nn.Dropout(dropout)
	def forward(self, inputs):
		embedded = self.drop(self.embedding(inputs))
		h_t = Variable(torch.zeros(inputs.size(0), self.hidden_size))
		c_t = Variable(torch.zeros(inputs.size(0), self.hidden_size))
		if self.use_cuda:
			h_t = h_t.cuda()
			c_t = c_t.cuda()
		for i, input_t in enumerate(embedded.chunk(embedded.size(1), dim=1)):
			h_t, c_t = self.lstm(torch.squeeze(input_t), (h_t, c_t))
		return h_t

class DecoderRnn(nn.Module):
	def __init__(self, vocab_size, hidden_size, dropout, use_cuda):
		super(DecoderRnn, self).__init__()
		self.hidden_size = hidden_size
		self.use_cuda = use_cuda
		self.vocab_size = vocab_size
		self.embedding = nn.Embedding(vocab_size, hidden_size)
		self.lstm = nn.LSTMCell(hidden_size, hidden_size)
		self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
		self.drop = nn.Dropout(dropout)
		self.linear = nn.Linear(hidden_size, vocab_size)
	def forward(self, inputs, hiddens, target_sequences, use_tf):
		embedded = self.drop(self.embedding(inputs))
		target_len = target_sequences.size(1)
		output= []
		h_t = hiddens
		c_t = Variable(torch.zeros(inputs.size(0), self.hidden_size))
		h_t2 = Variable(torch.zeros(inputs.size(0), self.hidden_size))
		c_t2 = Variable(torch.zeros(inputs.size(0), self.hidden_size))
		if self.use_cuda:
			c_t = c_t.cuda()
			h_t2 = h_t2.cuda()
			c_t2 = c_t2.cuda() 	
		batch_size = embedded.size(1)
		for i in range(target_len):
			h_t, c_t = self.lstm(torch.squeeze(embedded), (h_t, c_t))
			h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
			if use_tf:
				embedded = self.drop(self.embedding(target_sequences.permute(1,0)[i]))
			else:
				out = F.softmax(self.linear(h_t2)).max(1)[1]
				embedded = self.drop(self.embedding(out))
			output.append(F.softmax(self.linear(h_t2)))
		output = torch.stack(output)
		return output, h_t



