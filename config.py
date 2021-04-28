from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
import dataFormat
import filterData as fd
import numpy as np
import keras
import wget
import zipfile

class Config():
    def __init__(self, load=True):
        if load:
        	url = 'http://nlp.stanford.edu/data/glove.6B.zip'
        	filename = wget.download(url)
        	if not os.path.exists('glove.6B'):
        		os.mkdir('glove.6B')

        	with zipfile.ZipFile(filename,'r') as zf:
        		zf.extractall('glove.6B')
            self.load()

    def load(self):
        pre_data = fd.DataHandler()
        D_format = dataFormat.DataFormat()
        self.word_to_index, self.index_to_embed = pre_data.load_embedding_from_file(self.glove_name)
        if not os.path.exists(self.source_file_path):
            print("Generating data from corpus")
            pre_data.set_tokens(self.orig_text_folder_path)
            pre_data.write_tokens_to_file(self.source_file_path)
        all_text = D_format.load_token_data(self.source_file_path)
        tags = D_format.tags
        words = D_format.uni_word
        self.n_tags = len(tags)
        self.n_words = len(words)
        print('The number of unique words is {}'.format(self.n_words))
        self.embedding_matrix = np.zeros((self.n_words, self.EMBEDDING))

        for i, aw in enumerate(words):
            try:
                idx = self.word_to_index[aw]
                embd = list(self.index_to_embed[idx])
                if embd is not None:
                    self.embedding_matrix[i] = embd
                else:
                    self.embedding_matrix[i] = np.random.randn(self.EMBEDDING)
            except:
                print(aw)
        tag2idx = {t: i + 1 for i, t in enumerate(tags)}
        tag2idx["PAD"] = 0
        self.idx2tag = {i: w for w, i in tag2idx.items()}

        print("word to idx finished")

        self.X = [[self.word_to_index[w[0]] for w in s] for s in all_text]
        self.X = pad_sequences(maxlen=self.MAX_LEN, sequences=self.X, padding='post', value=0)

        self.Y = [[tag2idx[w[1]] for w in s] for s in all_text]
        self.Y = pad_sequences(maxlen=self.MAX_LEN, sequences=self.Y, padding="post", value=tag2idx["PAD"])
        self.Y = [to_categorical(i, num_classes=self.n_tags + 1) for i in self.Y]

        self.X_tr, self.X_te, self.y_tr, self.y_te = train_test_split(self.X, self.Y, test_size=self.test_size)

    BATCH_SIZE = 64  # Number of examples used in each iteration
    EPOCHS = 100  # Number of passes through entire dataset
    MAX_LEN = 50  # Max length of Sentence (in words)
    EMBEDDING = 50  # Dimension of word embedding vector

    orig_text_folder_path = 'tokenized'
    glove_name = 'glove.6B/glove.6B.{}d.txt'.format(EMBEDDING)
    source_file_path = 'token_text_with_label.txt'
    model_save_path = 'final_model_0421-50.h5'

    test_size = 0.1
