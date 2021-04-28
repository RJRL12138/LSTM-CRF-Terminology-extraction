from keras.initializers import Constant
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from sklearn_crfsuite.metrics import flat_classification_report
import numpy as np
from config import Config

# Load the configuration
config = Config()

# Build the model with Softmax layer as classification
input = Input(shape=(config.MAX_LEN,))
model = Embedding(input_dim=config.n_words, output_dim=config.EMBEDDING,
                  embeddings_initializer=Constant(config.embedding_matrix),trainable=False,
                  input_length=config.MAX_LEN)(input)
model = Bidirectional(LSTM(units=100, return_sequences=True,
                           recurrent_dropout=0.1))(model)
model = TimeDistributed(Dense(80, activation="relu"))(model)
model = Dense(40, activation='relu')(model)
out = Dense(3,activation='softmax')(model)
model = Model(input, out)

# Compile and train the model
model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(config.X_tr, np.array(config.y_tr), batch_size=config.BATCH_SIZE, epochs=config.EPOCHS, validation_split=0.1, verbose=2)

# Evaluate and get the summary fo prediction
pred_cat = model.predict(config.X_te)
pred = np.argmax(pred_cat, axis=-1)
y_te_true = np.argmax(config.y_te, -1)

pred_tag = [[config.idx2tag[i] for i in row] for row in pred]
y_te_true_tag = [[config.idx2tag[i] for i in row] for row in y_te_true]

report = flat_classification_report(y_pred=pred_tag, y_true=y_te_true_tag)
print(report)

# Save the model to a file
model.save(config.model_save_path)
