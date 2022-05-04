from gensim.models import Word2Vec
import numpy as np
import os
from textMG.datasets.dataset import Dataset
from textMG.configs.config import args
from textMG.utils.loggers import logger

class Get_embeddings:
    def __init__(self, path_word_embeddings=args.path_word_embeddings):
        self.dataset = Dataset()
        self.path_word_embeddings = path_word_embeddings
        self.w2v_model = self.load_model_word2v(self.path_word_embeddings)

    def load_model_word2v(self, path_word_embeddings, path_data_dir=args.path_data_dir):
        if not os.path.exists(path_word_embeddings):
            try:
                os.mkdir(path_word_embeddings)
            except Exception:
                logger.critical('this is an exception', exc_info=1)
                raise
        if len(os.listdir(path_word_embeddings)) == 0:
            try:
                sentences, _ = self.dataset.read_files(args.path_data_dir)
                self.train_word2vec(sentences, os.path.join(path_word_embeddings, "word2vec.model"))
            except Exception:
                logger.critical('this is an exception', exc_info=1)
                raise
        w2v_model = Word2Vec.load(os.path.join(path_word_embeddings, "word2vec.model"))
        return w2v_model

    def train_word2vec(self, sentences, path_save):
        # save and get model
        model = Word2Vec(sentences, vector_size=args.dim_size, window=4, min_count=1, workers=4)
        model.save(path_save)

    def word_idx(self, words):
        embedding_vector = self.w2v_model.wv[words]
        return embedding_vector

    def get_embeddings(self, path_data_dir=args.path_data_dir, vocab_file=args.vocab_file):
        vocab = self.dataset.load_vocab(args.path_data_dir, args.vocab_file)
        # use 0 to represent the non-existing words
        embedding_matrix = np.zeros((len(vocab) + 1, 50))
        for i, word in enumerate(vocab):
            try:
                embedding_vector = self.w2v_model.wv[word]
                embedding_matrix[i] = embedding_vector
            except KeyError:
                logger.critical('this is KeyError', exc_info=1)
                logger.critical("the excepted word is:{}".format(word))
                continue
        return embedding_matrix

if __name__ == "__main__":
    embed = Get_embeddings(args.path_word_embeddings)
    embedding_matrix = embed.get_embeddings()
    print(embedding_matrix.shape)