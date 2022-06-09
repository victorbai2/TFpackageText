import jieba
import random
import re
import os
from time import time
from collections import Counter
from textMG.configs.config import args, label_dict
from textMG.models.tokenization import FullTokenizer
import csv
from textMG.utils.loggers import logger
from typing import Callable, Dict, List, Union, Tuple


class FullTokenizerV2(FullTokenizer):
    def convert_tokens_to_ids(self, tokens: str) -> List[str]:
        return self.convert_by_vocab(self.vocab, tokens)

    def convert_by_vocab(self, vocab: Dict[str, int], items: str) -> List[str]:
        """Converts a sequence of [tokens|ids] using the vocab."""
        output = []
        for item in items:
            if item not in vocab:
                if item is ' ':
                    item = '[unused1]'  # space类用未经训练的[unused1]表示
                else:
                    item = '[UNK]'
            output.append(vocab[item])  # 剩余的字符是[UNK]
        return output


class Dataset:
    def __init__(self):
        self.value = None
        self.vocab = self.load_vocab(args.path_data_dir, args.vocab_file)

    def load_raw_dataset(self, path_rawdata: str) -> List[str]:
        files = [f for f in os.listdir(path_rawdata) if f.endswith(".csv")]
        data_raw = []
        for filename in files:
            data = []
            path = os.path.join(path_rawdata, filename)
            with open(path) as csv_file:
                rows = csv.reader(csv_file)
                next(rows, None)
                for row in rows:
                    if row[1]:
                        data.append([row[1], filename.split("_")[0]])
            data_raw.extend(data)
        return data_raw

    def clean_save(self, path_rawdata: str, path_save: str) -> None:
        """clean and save raw data for training"""
        t1 = time()
        if len(os.listdir(path_save)):
            logger.warning("data exists")
            return
        data_raw = self.load_raw_dataset(path_rawdata)
        # shuffle data
        random.seed(3)
        random.shuffle(data_raw)
        # here is to save data to disk
        suffix = 0
        path_ = os.path.join(path_save, 'news_data_' + str(suffix) + '.txt')
        f = open(path_, 'w')
        for i, line in enumerate(data_raw):
            try:
                x_, y_ = line[0].strip(), line[1]
                pattern = re.compile(r'[^\u4e00-\u9fa5]')
                x_ = re.sub(pattern, "", x_)
                if x_ and y_:
                    f.write(x_ + "|" + str(y_) + '\n')
            except Exception as e:
                logger.critical('this is exception', exc_info=1)
                logger.critical("the excepted line is:{}".format(line))
                continue
            if i != 0 and not i % 10000:
                f.close()
                suffix += 1
                path_ = os.path.join(path_save, 'news_data_' + str(suffix) + '.txt')
                f = open(path_, 'w')
        f.close()
        logger.info("saved cleaned data to", path_save)
        logger.info("timed used: {}s".format(time() - t1))

    def read_single_file(self, file_path: str) -> Tuple[List[str]]:
        """read single file with single category"""
        content = []
        label = []
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    x, y = line.split("|")
                    x = jieba.lcut(x.strip())
                    y = y.strip()
                    if x and y:
                        content.append(x)
                        label.append(y)
                except Exception as e:
                    logger.critical('this is exception', exc_info=1)
                    logger.critical("the excepted line is:{}".format(line))
                    continue
        return content, label

    def read_files(self, dirname: str, suffix=".txt") -> Tuple[List[str]]:
        """read files"""
        contents = []
        labels = []
        files = [f for f in os.listdir(dirname) if f.endswith(suffix)]
        for filename in files:
            content, label = self.read_single_file(os.path.join(dirname, filename))
            contents.extend(content)
            labels.extend(label)
        return contents, labels

    def build_vocab(self, data_dir: str, vocab_file: str, vocab_size: int=None) -> None:
        """build the vocabularies and save it to disk"""
        contents, _ = self.read_files(data_dir)
        all_data = [i for j in contents for i in j]
        if vocab_size:
            vocab = Counter(all_data).most_common(vocab_size)
            words, _ = list(zip(*vocab))
            # add <PAD> mark at the first line
            words = ['<PAD>'] + list(words)
        else:
            words = [i[0] for i in Counter(all_data).items() if i[1] > 1]
            # add <PAD> mark at the first line
            words = ['<PAD>'] + words

        with open(os.path.join(vocab_file, "vocab.txt"), mode='w') as f:
            f.write('\n'.join(words))

    def load_vocab(self, data_dir: str, vocab_file: str) -> Dict[str, int]:
        vocab = {}
        if len(os.listdir(vocab_file)) == 0:
            self.build_vocab(data_dir, vocab_file, vocab_size=40000)
        with open(os.path.join(vocab_file, "vocab.txt"), 'r') as f:
            for i, w in enumerate(f):
                w = w.strip()
                vocab[w] = i
        logger.info("vocabulary size:{}".format(len(vocab)))
        return vocab

    def load_stopwords(self, path_stopwords: str) -> List[str]:
        stopwords = []
        with open(path_stopwords, 'r') as f:
            for i in f:
                w = i.strip()
                stopwords.append(w)
        return stopwords

    def tokenizer(self, lines: List[str], vocab: Dict[str, int]) -> List[int]:
        tokenized_data = [vocab[w] if w in vocab else 0 for w in lines]
        return tokenized_data

    def bert_tokenizer(self, line: str, F_tokenizer: Callable) -> List[int]:
        tokenized_data = F_tokenizer.convert_tokens_to_ids(line)
        return tokenized_data

    def pad_sequences(self, x: List[Union[int, str]], max_len: int) -> List[Union[int, str]]:
        if len(x) < max_len:
            x = x+[0]*(max_len-len(x))
        else:
            x = x[:max_len]
        return x

    def one_hot(self, x: int) -> List[int]:
        scalar = [0] * len(label_dict)
        scalar[x] = 1
        return scalar

    def process_data(self, data_dir: str, vocab_file: str, path_stopwords: str, n_examples=None) -> Tuple[List[int]]:
        """
        # 下列这些都是一个代码匹配一个字符（即代码，字符一一对应才能匹配成功）
        # 代码 功能
        # . 匹配任意1个字符（除了\n）
        # [ ] 匹配[ ]中列举的字符
        # \d 匹配数字，即0-9
        # \D 匹配非数字，即不是数字
        # \s 匹配空白，即 空格，tab键
        # \S 匹配非空白
        # \w 匹配非特殊字符，即a-z、A-Z、0-9、_、汉字
        # \W 匹配特殊字符，即非字母、非数字、非汉字、非_
        """
        t1=time()
        files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
        vocab = self.vocab
        stopwords = self.load_stopwords(path_stopwords)
        x = []
        y = []
        num = 0
        for filename in files:
            with open(os.path.join(data_dir, filename), 'r') as f:
                for line in f:
                    try:
                        x_, y_ = line.strip().split("|")
                        x_ = jieba.lcut(x_.strip())
                        x_ = list(filter(lambda x: len(x) > 1, x_))  # filer out the words with len < 2
                        x_ = list(filter(lambda x: x not in stopwords, x_))  # filter the stopwords
                        x_ = self.tokenizer(x_, vocab)
                        y_ = label_dict[y_]
                        if len(x_) > 2 and y_:
                            x_ = self.pad_sequences(x_, args.max_len)
                            y_ = self.one_hot(y_)
                            x.append(x_)
                            y.append(y_)
                            num += 1
                            if n_examples and num == n_examples:
                                return x, y
                    except Exception as e:
                        logger.critical('this is exception', exc_info=1)
                        logger.critical("the excepted line is:{}".format(line))
                        continue
        logger.info("time used to process the all data: {}s".format(time()-t1))
        return x, y

    def inquiry_process_pred(self, inquiries: List[str]) -> List[int]:
        vocab = self.vocab
        stopwords = self.load_stopwords(args.path_stopwords)
        if len(inquiries)==0:
            return "Please provide your inquiry"
        else:
            tokens=[]
            for i in inquiries:
                token = self.process_single_inquiry(i, vocab, stopwords)
                tokens.append(token)
            return tokens

    def process_single_inquiry(self, inquiry: str, vocab: Dict[str, int], stopwords: List[str]) -> List[int]:
        x_ = jieba.lcut(inquiry.strip())
        x_ = list(filter(lambda x: len(x) >= 1, x_))  # filer out the words with len < 2
        x_ = list(filter(lambda x: x not in stopwords, x_))  # filter the stopwords
        x_ = self.tokenizer(x_, vocab)
        if x_:
            x_ = self.pad_sequences(x_, args.max_len)
        return x_

    def process_data_pretrained(self, data_dir: str, vocab_file: str, path_stopwords: str, max_len: int,
                                is_token_b: bool=False, n_examples=None) -> Tuple[List]:
        """
        load and tokenize data for pretrained embedding
        """
        t1=time()
        files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
        F_tokenizer = FullTokenizerV2(vocab_file=vocab_file, do_lower_case=args.do_lower_case)
        stopwords = self.load_stopwords(path_stopwords)
        input_ids=[]
        input_masks=[]
        input_type_ids=[]
        y_output = []
        num = 0
        for filename in files:
            with open(os.path.join(data_dir, filename), 'r') as f:
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
                                x = self.pad_sequences(x, max_len)
                                mask = self.pad_sequences(mask, max_len)
                                input_type = [0] * max_len
                                input_ids.append(x)
                                input_masks.append(mask)
                                input_type_ids.append(input_type)
                                y = self.one_hot(y)
                                y_output.append(y)
                                num += 1
                                if n_examples and num == n_examples:
                                    return input_ids, input_masks, input_type_ids, y_output
                        except Exception as e:
                            logger.critical('this is exception', exc_info=1)
                            logger.critical("the excepted line is:{}".format(line))
                            continue
        logger.info("time used to process the all data: {}s".format(time()-t1))
        return input_ids, input_masks, input_type_ids, y_output


if __name__ == '__main__':
    dataset = Dataset()
    # dataset.clean_save(args.path_rawdata, args.path_data_dir)
    x, y = dataset.process_data(args.path_data_dir, args.vocab_file, args.path_stopwords)
    print("y:", y[:20])
