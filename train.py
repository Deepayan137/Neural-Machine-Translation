import torch 
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
import argparse
import os
import numpy as np
from model import EncoderRnn, DecoderRnn
from utils import Lang, split_data
import pdb
import random
from loader import LanguageDataset, Tokenize, EncodeSeq, Pad, ToTensor

teacher_forcing_ratio=0.5
def train(batch_input_sequences, batch_target_sequences, 
	criterion, encoder, decoder, lr,  eos_tok, use_cuda):
	encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
	decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()
	use_tf = True if random.random() < teacher_forcing_ratio else False
	if use_cuda:
		batch_input_sequences = (batch_input_sequences).cuda()
		batch_target_sequences = (batch_target_sequences).cuda()
	batch_input_sequences = Variable(batch_input_sequences)
	batch_target_sequences = Variable(batch_target_sequences)
	encoder.train(); decoder.train()
	encoder_hiddens = encoder(batch_input_sequences)
	decoder_inputs = np.zeros((len(batch_input_sequences), 1))
	for i in range(len(batch_input_sequences)):
		decoder_inputs[i] = eos_tok
	decoder_inputs = torch.from_numpy(decoder_inputs).long()
	if use_cuda:
		decoder_inputs = decoder_inputs.cuda()
	decoder_inputs = Variable(decoder_inputs)
	decoder_hiddens = encoder_hiddens
			
	decoder_outputs, decoder_hiddens= decoder(decoder_inputs, decoder_hiddens, batch_target_sequences, use_tf)
	loss = 0
	for i, d_o in enumerate(decoder_outputs):
		loss += criterion(d_o, batch_target_sequences.permute(1, 0)[i])
	loss.backward()
	encoder_optimizer.step()
	decoder_optimizer.step()
	return loss.data[0]
		
def validate(batch_input_sequences, batch_target_sequences, 
	criterion, encoder, decoder, lr,  eos_tok, use_cuda):
	if use_cuda:
		batch_input_sequences = (batch_input_sequences).cuda()
		batch_target_sequences = (batch_target_sequences).cuda()
	batch_input_sequences = Variable(batch_input_sequences)
	batch_target_sequences = Variable(batch_target_sequences)
	encoder.eval(); decoder.eval()
	encoder_hiddens = encoder(batch_input_sequences)
	decoder_inputs = np.zeros((len(batch_input_sequences), 1))
	for i in range(len(batch_input_sequences)):
		decoder_inputs[i] = eos_tok
	decoder_inputs = torch.from_numpy(decoder_inputs).long()
	if use_cuda:
		decoder_inputs = decoder_inputs.cuda()
	decoder_inputs = Variable(decoder_inputs)
	decoder_hiddens = encoder_hiddens
			
	decoder_outputs, decoder_hiddens= decoder(decoder_inputs, decoder_hiddens, batch_target_sequences, 0)
	loss = 0
	# pdb.set_trace()
	for i, d_o in enumerate(decoder_outputs):
		loss += criterion(d_o, batch_target_sequences.permute(1, 0)[i])
	return loss.data[0]
if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('-max_len', type=int, default=100)
	ap.add_argument('-vocab_size', type=int, default=100)
	ap.add_argument('-batch_size', type=int, default=128)
	ap.add_argument('-hidden_dim', type=int, default=300)
	ap.add_argument('-dropout', type=float, default=0.3)
	ap.add_argument('-nb_epoch', type=int, default=25)
	ap.add_argument('-learning_rate', type=int, default=0.01)
	ap.add_argument('-log_step', type=int, default=5)
	args = vars(ap.parse_args())

	MAX_LEN = args['max_len']
	VOCAB_SIZE = args['vocab_size']
	BATCH_SIZE = args['batch_size']
	HIDDEN_DIM = args['hidden_dim']
	NB_EPOCH = args['nb_epoch']
	LEARNING_RATE = args['learning_rate']
	DROPOUT = args['dropout']
	def save(model_ft, filename):
		save_filename = filename
		torch.save(model_ft, save_filename)
		print('Saved as %s' % save_filename)
	# creating vocabulary
	def create_vocab(filename):
		sentences = LanguageDataset(lang_file=filename, root_dir='data')
		lang_obj = Lang(sentences, VOCAB_SIZE)
		return lang_obj.word2index()
	# pair wise matching
	def pairwise(input_sequences, target_sequences, indices):
		inp = torch.stack([input_sequences[indices[i]] for i in range(len(indices))])
		tar = torch.stack([target_sequences[indices[i]] for i in range(len(indices))])
		return inp, tar

	vocab_eng = create_vocab('europarl-v8.fi-en.en')
	vocab_fin = create_vocab('europarl-v8.fi-en.fi')

	#Loading Data
	input_sequences = LanguageDataset(lang_file='europarl-v8.fi-en.en', root_dir='data',
									transform=transforms.Compose([Tokenize(),EncodeSeq(vocab_eng),
																	Pad(MAX_LEN),
																	ToTensor()]))
	target_sequences = LanguageDataset(lang_file='europarl-v8.fi-en.fi', root_dir='data',
									transform=transforms.Compose([Tokenize(),EncodeSeq(vocab_fin),
																	Pad(MAX_LEN),
																	ToTensor()]))
	indices = list(range(len(input_sequences)))
	split_index = int(0.8*len(indices))
	np.random.shuffle(indices)
	train_idx, valid_idx = indices[:split_index], indices[split_index:]

	
	train_input_sequences, train_target_sequences = pairwise(input_sequences, target_sequences, train_idx)
	val_input_sequences, val_target_sequences = pairwise(input_sequences, target_sequences, valid_idx)
	train_size = len(train_idx)
	val_size = len(valid_idx)
	# pdb.set_trace()
	eos_tok = vocab_fin['$']
	use_gpu = torch.cuda.is_available()
	print("loading encoder and decoder models")
	use_gpu = 0
	encoder = EncoderRnn(len(vocab_eng), HIDDEN_DIM, DROPOUT, use_gpu)
	decoder = DecoderRnn(len(vocab_fin), HIDDEN_DIM, DROPOUT, use_gpu)
	if use_gpu:
		print("using GPU")
		encoder = encoder.cuda()
		decoder = decoder.cuda()
	criterion = nn.CrossEntropyLoss()
	total_step = (train_size/BATCH_SIZE) +1
	
	for epoch in  range(NB_EPOCH):
		step = 0
		try:
			for i in range(0, train_size, BATCH_SIZE):
				step+=1
				num_samples = min(BATCH_SIZE, train_size - i)
				train_batch_input_sequences = train_input_sequences[i:i+num_samples]
				train_batch_target_sequences = train_target_sequences[i:i+num_samples]
				train_loss = train(train_batch_input_sequences, train_batch_target_sequences, 
							criterion, encoder, decoder,
							LEARNING_RATE, eos_tok, use_gpu)
				
				val_loss=0.0; count = 0
				for j in range(0, val_size, BATCH_SIZE):
					count+=1	
					num_samples = min(BATCH_SIZE, val_size - j)
					val_batch_input_sequences = val_input_sequences[j:j+num_samples]
					val_batch_target_sequences = val_target_sequences[j:j+num_samples]
					val_loss += validate(val_batch_input_sequences, val_batch_target_sequences, 
								criterion, encoder, decoder,
								LEARNING_RATE, eos_tok, use_gpu)
				if step % args['log_step']:
					print('Epoch [%d/%d], Step [%d/%d], Train Loss: %.4f \n'%(epoch+1, NB_EPOCH, step, 
					total_step, train_loss))
					print('Validation Loss %.4f'%(val_loss/float(count)))
		except KeyboardInterrupt:
			print("Saving before quit...")
			save(encoder, 'encoder_[%d/%d].pkl'%(step, total_step))
			save(decoder, 'decoder_[%d/%d].pkl'%(step, total_step))
	save(encoder, 'encoder.pkl')
	save(decoder, 'decoder.pkl')