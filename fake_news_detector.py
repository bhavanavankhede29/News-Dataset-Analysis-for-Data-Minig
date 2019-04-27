import pandas as pd
import numpy as np
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.utils import to_categorical
import pickle
from keras import Sequential, Model, Input
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Flatten, Dense, GlobalAveragePooling1D, Dropout, LSTM, CuDNNLSTM, RNN, SimpleRNN, Conv2D, GlobalMaxPooling1D
from keras import callbacks


MAX_NB_WORDS=50000 #dictionary size
MAX_SEQUENCE_LENGTH=1500 #max word length of each individual article
EMBEDDING_DIM=300 #dimensionality of the embedding vector (50, 100, 200, 300)
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')


def load_newsDataset():
    #load training data and put into arrays
    df = pd.read_csv('artciles.csv', encoding='utf8') # be sure to point to wherever you put your file
    training_data = df['Body'].values.tolist() #'text' column contains articles
    training_labels = df['Label'].values.tolist() #'label' column contains labels

    #Randomly shuffle data and labels together
    combo = list(zip(training_data, training_labels))
    random.shuffle(combo)
    training_data, training_labels = zip(*combo)
    del df #clear up memory

    return np.asarray(training_data).tolist(), np.asarray(training_labels).tolist()

training_data, training_labels = load_newsDataset()

#print( training_labels)


def tokenize_trainingdata(texts, labels):
    tokenizer.fit_on_texts(texts)
    pickle.dump(tokenizer, open('tokenizer.p', 'wb'))

    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(labels, num_classes=len(set(labels)))

    return data, labels, word_index

####and run it
X, Y, word_index = tokenize_trainingdata(training_data, training_labels)  

train_data = X[:int(len(X)*0.9)]
train_labels = Y[:int(len(X)*0.9)]
test_data = X[int(len(X)*0.9):int(len(X)*0.95)]
test_labels = Y[int(len(X)*0.9):int(len(X)*0.95)]
valid_data = X[int(len(X)*0.95):]
valid_labels = Y[int(len(X)*0.95):]



def load_embeddings(word_index, embeddingsfile='glove.6B.%id.txt' %EMBEDDING_DIM):
    embeddings_index = {}
    f = open(embeddingsfile, 'r', encoding='utf8')
    for line in f:
        #here we parse the data from the file
        values = line.split(' ') #split the line by spaces
        word = values[0] #each line starts with the word
        coefs = np.asarray(values[1:], dtype='float32') #the rest of the line is the vector
        embeddings_index[word] = coefs #put into embedding dictionary
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    return embedding_layer
    
###and build the embedding layer
embedding_layer = load_embeddings(word_index)


def baseline_model(sequence_input, embedded_sequences, classes=2):
    x = Conv1D(64, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(256, 2, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    preds = Dense(classes, activation='softmax')(x)

    model = Model(sequence_input, preds)
    return model


###put embedding layer into input of the model
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

model = baseline_model(sequence_input, embedded_sequences, classes=2)

model.compile(loss='categorical_crossentropy',
              optimizer='adamax',
              metrics=['acc'])

#print(model.summary())

model.fit(train_data, train_labels,
          validation_data=(valid_data, valid_labels),
          epochs=20, batch_size=64)

model.evaluate(test_data, test_labels)
