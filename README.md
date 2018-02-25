# Neural-Machine-Translation
A naive approach towards Machine Translation using a sequence to sequence model. @Pytorch

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the dataset ([Europarl's Parallel Corpus](https://drive.google.com/drive/folders/19JFnTlK59UanTNf9g6yp3MOZ4teooHST)) and move the dataset to a subfolder data/

Feel free to change the default dataset to anyone of your own. Just don't forget to modify the code!

## Training

To train a model, run `train.py`

```bash
python train.py 
```

## Usage 
```
Pytorch Seq2seq encoder decoder training for MT

optional arguments:
  -h, --help            Thow this help message and exit
  -nb_epoch            	number of total epochs to run
  -batch_size			mini-batch size (default: 128)
  -learning_rate		learning rate default set to 0.01
  -vocab_size			size of vocabulary default set to 10000
  -hidden_dim 			size of hidden layer default set to 300
  -dropout				dropout probability default set to 0.3
```
