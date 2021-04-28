import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from config import Config


config = Config()

# Read an existing model from file then evaluate with this model
model = keras.models.load_model('final_model_0419-{}.h5'.format(config.EMBEDDING),
                              custom_objects={'CRF': CRF, 'crf_loss': crf_loss, 'crf_viterbi_accuracy': crf_accuracy})
model.summary()
example = 'Natural Language is a term .'.lower().split(' ')
ex_idx = []
for word in example:
    if config.word_to_index.__contains__(word):
        ex_idx.append(config.word_to_index[word])
    else:
        ex_idx.append(0)

ex = pad_sequences(maxlen=50, sequences=[ex_idx], padding='post', value=0)
ex_pred = model.predict(ex)
ex_re = np.argmax(ex_pred, axis=-1)
ex_tag = [[config.idx2tag[i] for i in row] for row in ex_re]

for i in range(0,len(example)):
    print(example[i]+':'+ex_tag[0][i])


