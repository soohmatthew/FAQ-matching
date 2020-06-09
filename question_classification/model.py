'''
Written by Austin Walters
Last Edit: January 2, 2019
For use on austingwalters.com

A CNN to classify a sentence as one 
of the common sentance types:
Question, Statement, Command, Exclamation

Heavily Inspired by Keras Examples: 
https://github.com/keras-team/keras
'''

from __future__ import print_function

import os
import sys

import numpy as np
import keras

from question_classification.sentence_types import load_encoded_data
from question_classification.sentence_types import encode_data, import_embedding, encode_data_for_prediction
from question_classification.sentence_types import get_custom_test_comments

from keras.preprocessing import sequence
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D

from keras.preprocessing.text import Tokenizer

class Question_classifier():
    def __init__(self, model_name = "model/cnn-classifier", embedding_name = "data/default", pos_tags_flag = True, maxlen = 500):
        self.model_name = model_name
        self.embedding_name = embedding_name
        self.load_model_flag = os.path.isfile(self.model_name+".json")
        self.model = None
        # Add parts-of-speech to data
        self.pos_tags_flag = pos_tags_flag
        self.maxlen = maxlen

    def train_model(self):
        # Model configuration
        batch_size = 64
        embedding_dims = 75
        filters = 100
        kernel_size = 5
        hidden_dims = 350
        epochs = 2

        x_train, x_test, y_train, y_test = load_encoded_data(data_split=0.8,
                                                        embedding_name=self.embedding_name,
                                                        pos_tags=self.pos_tags_flag)

        word_encoding, category_encoding = import_embedding(self.embedding_name)

        max_words   = len(word_encoding) + 1
        num_classes = np.max(y_train) + 1

        print(max_words, 'words')
        print(num_classes, 'classes')

        print('Pad sequences (samples x time)')
        x_train = sequence.pad_sequences(x_train, maxlen=self.maxlen)
        x_test = sequence.pad_sequences(x_test, maxlen=self.maxlen)

        print('Convert class vector to binary class matrix '
            '(for use with categorical_crossentropy)')
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        print('Constructing model!')

        self.model = Sequential()

        self.model.add(Embedding(max_words, embedding_dims,
                            input_length=self.maxlen))
        
        self.model.add(Dropout(0.2))
        
        self.model.add(Conv1D(filters, kernel_size, padding='valid',
                        activation='relu', strides=1))
        
        self.model.add(GlobalMaxPooling1D())
        
        self.model.add(Dense(hidden_dims))
        self.model.add(Dropout(0.2))
        self.model.add(Activation('relu'))
        
        self.model.add(Dense(num_classes))
        self.model.add(Activation('softmax'))
        
        self.model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        
        self.model.fit(x_train, y_train, batch_size=batch_size,
                epochs=epochs, validation_data=(x_test, y_test))

        model_json = self.model.to_json()
        with open(self.model_name + ".json", "w") as json_file:
            json_file.write(model_json)
        
        # serialize weights to HDF5
        self.model.save_weights(self.model_name + ".h5")
        print("Saved model to disk")

    def load_model(self):
        print('Loading model!')

        # load json and create model
        json_file = open(self.model_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        
        # load weights into new model
        self.model.load_weights(self.model_name + ".h5")
        print("Loaded model from disk")
        
        # evaluate loaded model on test data
        self.model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

        print("Model loaded")
    
    def predict(self, test_comments):
        if self.model is not None:
            assert isinstance(test_comments, str), "Please input a string"
            test_comments = [test_comments]
            test_comments_category = [""]

            x_test, _, _, _ = encode_data(test_comments, test_comments_category,
                                            data_split=1.0,
                                            embedding_name= self.embedding_name,
                                            add_pos_tags_flag= self.pos_tags_flag)

            x_test = sequence.pad_sequences(x_test, maxlen=self.maxlen)
            # Show predictions
            predictions = self.model.predict(x_test, batch_size=1, verbose= 0)
            return predictions[0].argmax(axis=0) == 1
        else:
            raise Exception('Please load model first')

if __name__ == "__main__":
    qc = Question_classifier()
    qc.load_model()
    print(qc.predict("Is this a question?"))
    print(qc.predict("Mcdonalds is the best."))