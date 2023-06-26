from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 300
VALIDATION_SPLIT = 0.2
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
def Detect_sentiment(sentences):
    resp = list()
    test_sequences = tokenizer.texts_to_sequences(sentences)
    test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    nn_output = model.predict(test_data)
    index_to_label_dict = {0: 0.0, 1: 1.0, 2: -1.0}
    i=0
    for idx in np.argmax(nn_output, axis=1):
        if index_to_label_dict[idx] == -1 : sentiment = 'Negative'
        elif index_to_label_dict[idx] == 0 : sentiment = 'Neutral'
        elif index_to_label_dict[idx] == 1 : sentiment = 'Positive'
        resp.append(sentiment)
        i = i + 1
    return resp
def Execute_model(path):
    model = keras.models.load_model(path)
    return model

#INICIALIZAR VALORES PARA QUE FUNCIONE EL TOKENIZER
texts = []
labels_index = {}
labels = []

input_df = pd.read_csv('Twitter_Data.csv')
review_df1 = input_df[['clean_text','category']]
review_df = review_df1.sample(frac=1, random_state=42)
review_df = review_df.dropna()

texts = review_df['clean_text'].values.tolist()
labels = []
labels_text = []
labels_text_unique = review_df.category.unique().tolist()
labels_text = review_df['category'].values.tolist()

idxCounter = 0
for label in labels_text_unique:
    labels_index[label] = idxCounter
    idxCounter = idxCounter + 1;

idxCounter = 0
for label in labels_text:
    labels.append(labels_index[label])
    idxCounter = idxCounter + 1;

#INICIALIZAR VALORES PARA QUE FUNCIONE EL TOKENIZER
from tensorflow.keras.layers import Embedding
from tensorflow.keras.initializers import Constant
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))

indices = np.arange(data.shape[0])
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

model = keras.models.load_model('SentimentModelCNNv2.h5', compile=True)
