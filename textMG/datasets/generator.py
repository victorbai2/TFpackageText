import pickle
import os
import jieba
from time import time
from textMG.configs.config import args, label_dict
from textMG.datasets.dataset import Dataset, FullTokenizerV2
from textMG.datasets.data_loader import load_data
from textMG.utils.loggers import logger


class Generator:
	def __init__(self, dataset=Dataset, is_pretrained=False):
		self.dataset = dataset()
		self.is_pretrained = is_pretrained
		if not self.is_pretrained:
			self.vocab = self.dataset.load_vocab(args.path_data_dir, args.vocab_file)
			self.stopwords = self.dataset.load_stopwords(args.path_stopwords)
			self.tokenizer = self.dataset.tokenizer
			self.iter = self.iterator(args.path_data_dir)
			self.input = 'x_input'
			self.output = 'y_output'
		else:
			self.bert_tokenizer = self.dataset.bert_tokenizer
			self.iter = self.iterator_pretrained(args.path_data_dir, args.pretrained_vocab_file, max_len=args.max_len)
			self.input_ids = 'input_ids'
			self.input_masks = 'input_masks'
			self.input_type_ids = 'input_type_ids'
			self.y_output = 'y_output'

	def get_next_patch(self, batch=None):
		if batch:
			try:
				while True:
					if not self.is_pretrained:
						x_input = []
						y_output = []
						for _ in range(batch):
							x, y = next(self.iter)
							x_input.append(x)
							y_output.append(y)
						yield {
							self.input: x_input,
							self.output: y_output}
					else:
						input_ids = []
						input_masks = []
						input_type_ids = []
						y_output = []
						for _ in range(batch):
							x, mask, input_type, y = next(self.iter)
							input_ids.append(x)
							input_masks.append(mask)
							input_type_ids.append(input_type)
							y_output.append(y)
						yield {
								self.input_ids: input_ids,
								self.input_masks: input_masks,
								self.input_type_ids: input_type_ids,
								self.y_output: y_output
						}
			except EOFError as e:
				logger.critical("EOFError occurred", exc_info=1)
				raise e
		else:
			try:
				while True:
					if not self.is_pretrained:
						x_input, y_output = next(self.iter)
						yield {
								self.input: x_input,
								self.output: y_output}
					else:
						x, mask, input_type, y = next(self.iter)
						yield {
								self.input_ids: x,
								self.input_masks: mask,
								self.input_type_ids: input_type,
								self.y_output: y
							}
			except EOFError as e:
				logger.critical("EOFError occurred", exc_info=1)
				raise e

	def iterator(self, path_data_dir):
		files = [f for f in os.listdir(path_data_dir) if f.endswith(".txt")]
		for filename in files:
			with open(os.path.join(path_data_dir, filename), 'r') as f:
				for line in f:
					try:
						x_, y_ = line.strip().split("|")
						x_ = jieba.lcut(x_.strip())
						x_ = list(filter(lambda x: len(x) > 1, x_))  # filer out the words with len < 2
						x_ = list(filter(lambda x: x not in self.stopwords, x_))  # filter the stopwords
						x_ = self.tokenizer(x_, self.vocab)
						y_ = label_dict[y_]
						if len(x_) > 2 and y_:
							x_ = self.dataset.pad_sequences(x_, args.max_len)
							y_ = self.dataset.one_hot(y_)
							yield x_, y_
					except Exception as e:
						logger.critical('this is exception', exc_info=1)
						logger.critical("the excepted line is:{}".format(line))
						continue

	def iterator_pretrained(self, path_data_dir, vocab_file, max_len, is_token_b=False):
		files = [f for f in os.listdir(path_data_dir) if f.endswith(".txt")]
		F_tokenizer = FullTokenizerV2(vocab_file=vocab_file, do_lower_case=args.do_lower_case)
		for filename in files:
			with open(os.path.join(path_data_dir, filename), 'r') as f:
				if not is_token_b:
					for line in f:
						try:
							x_, y_ = line.strip().split("|")
							# x_ = jieba.lcut(x_.strip())
							# x_ = list(filter(lambda x: len(x) > 1, x_))  # filer out the words with len < 2
							# x_ = list(filter(lambda x: x not in stopwords, x_))  # filter the stopwords
							x = ["[CLS]"]
							for i in x_:
								x.append(i)
							x.append("[SEP]")
							x = self.bert_tokenizer(x, F_tokenizer)
							mask = [1] * len(x)
							y = label_dict[y_]
							if len(x) > 0 and y:
								x = self.dataset.pad_sequences(x, max_len)
								mask = self.dataset.pad_sequences(mask, max_len)
								input_type = [0] * max_len
								y = self.dataset.one_hot(y)
								yield x, mask, input_type, y
						except Exception as e:
							logger.critical('this is exception', exc_info=1)
							logger.critical("the excepted line is:{}".format(line))
							continue

	def data_init(self):
		mnist = load_data(args.data_load_dir)
		if self.dataset == 'train':
			return mnist.train.images, mnist.train.labels
		elif self.dataset == 'eval':
			return mnist.test.images, mnist.test.labels
		else:
			raise ValueError

	def load_eval_data(self, path_eval_data):
		if os.path.exists(path_eval_data):
			with open(path_eval_data, 'rb') as f:
				mnist = pickle.load(f)
		else:
			self.pickle_dump(path_eval_data)
		return mnist

	def pickle_dump(self, full_path):
		x, y = self.data_init()
		with open(full_path, 'wb') as f:
			for line in zip(x, y):
				pickle.dump(line, f)
if __name__ == '__main__':
	t1=time()
	generator = Generator()
	iter = generator.get_next_patch(batch=5)
	for _ in range(4):
		output=next(iter)
		print("y :", output['y_output'])
	print("time used :{}s".format(time()-t1))
