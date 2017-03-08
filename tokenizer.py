#!/usr/bin/env python

import sys
import random
import itertools
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, SimpleRNN, GRU, Dropout, Input, Flatten, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support



class Reader(object):

    def __init__(self, input):
        self.input = input


    @staticmethod
    def encode(x, n):
        result = np.zeros(n)
        result[x] = 1
        return result
    

class IOBReader(Reader):
    
    def __init__(self, input):
        super(IOBReader, self).__init__(input)
        self.char2index = {}
        self.index2char = {}
        self.label2index = {}
        self.index2label = {}

        self.X_train = []
        self.X_test = []

        self.y_train = []
        self.y_test = []

    
    def read(self, delimiter='\t'):
        chars = set()
        labels = set()
        X = []
        y = []
        examples = []
        for line in self.input:
            char, label = (ch.strip() for ch in line.split(delimiter))
            chars.add(char)
            labels.add(label)
            examples.append((char, label))
        
        for i, c in enumerate(chars):
            self.char2index[c] = i
            self.index2char[i] = c

        for i, l in enumerate(labels):
            self.label2index[l] = i
            self.index2label[i] = l

        # build sequences
        sequence_X = []
        sequence_y = []
        for char, label in examples:
            if label == 'B-S' and sequence_y.count('B-S') > random.randint(1, 4):
                X.append(sequence_X)
                y.append(sequence_y)
                sequence_X = []
                sequence_y = []
            sequence_X.append(char)
            sequence_y.append(label)


        X_enc = [[self.char2index[c] for c in x] for x in X]
        self.max_label = max(self.label2index.values()) + 1

        y_enc = [[0] * (self.maxlen - len(ey)) + [self.label2index[c] for c in ey] for ey in y]
        #one_hot
        y_enc = [[Reader.encode(c, self.max_label) for c in ey] for ey in y_enc]

        X_enc = pad_sequences(X_enc, maxlen=self.maxlen)
        y_enc = pad_sequences(y_enc, maxlen=self.maxlen)
    
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_enc, y_enc, random_state=42)


class SequencesIOBReader(IOBReader):

    @staticmethod
    def extractWindows(dataset, labels, window_size):
        d = []
        l = []
        for i in range(len(dataset)-window_size+1):
            d.append(dataset[i:i+window_size])
            l.append(labels[i:i+window_size])
        return np.asarray(d), np.asarray(l)

    
    def read(self, delimiter='\t'):
        X = []
        y = []

        for line in self.input:
            char, label = (ch.strip() for ch in line.split(delimiter))
            X.append(char)
            y.append(label)
        
        for i, c in enumerate(set(X)):
            self.char2index[c] = i
            self.index2char[i] = c

        for i, l in enumerate(set(y)):
            self.label2index[l] = i
            self.index2label[i] = l

        X, y = SequencesIOBReader.extractWindows(X, y, 5)

        self.maxlen = max([len(x) for x in X])

        X_enc = [[self.char2index[c] for c in x] for x in X]
        self.max_label = max(self.label2index.values()) + 1

        #one_hot
        y_enc = [[Reader.encode(self.label2index[c], self.max_label) for c in ey] for ey in y]

        X_enc = pad_sequences(X_enc, maxlen=self.maxlen)
        y_enc = pad_sequences(y_enc, maxlen=self.maxlen)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_enc, y_enc, random_state=42)

        
class WindowsIOBWriter(object):
    @staticmethod
    def decode(elems):
        return [np.where(l==1)[0][0] for l in elems]


    def write(self, y_test):
        return WindowsIOBWriter.decode(y_test)


class SequencesIOBWriter(object):
    @staticmethod
    def decode(elems):
        toRet = []
        for sentence in elems:
            toRet.append([np.where(el==1)[0][0] for el in sentence])
        return toRet
    
    def write(self, y_test, y_pred):
        y_test = list(itertools.chain.from_iterable(SequencesIOBWriter.decode(y_test)))
        y_pred = list(itertools.chain.from_iterable(y_pred))
        return y_test, y_pred


class WindowsIOBReader(IOBReader):

    @staticmethod
    def extractWindows(dataset, labels, window_size):
        d = []
        l = []
        for i in range(len(dataset)-window_size+1):
            d.append(dataset[i:i+window_size])
            l.append(labels[window_size+i-3])
        return np.asarray(d), np.asarray(l)

    
    def read(self, delimiter='\t'):
        X = []
        y = []

        for line in self.input:
            char, label = (ch.strip() for ch in line.split(delimiter))
            X.append(char)
            y.append(label)
        
        for i, c in enumerate(set(X)):
            self.char2index[c] = i
            self.index2char[i] = c

        for i, l in enumerate(set(y)):
            self.label2index[l] = i
            self.index2label[i] = l

        X, y = WindowsIOBReader.extractWindows(X, y, 5)

        self.maxlen = max([len(x) for x in X])

        X_enc = [[self.char2index[c] for c in x] for x in X]
        self.max_label = max(self.label2index.values()) + 1

        y_enc = [Reader.encode(self.label2index[c], self.max_label) for c in y]
        X_enc = pad_sequences(X_enc, maxlen=self.maxlen)
        y_enc = pad_sequences(y_enc)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_enc, y_enc, random_state=42)



class Tokenizer(object):
    def __init__(self, reader=WindowsIOBReader, input=sys.stdin):
        self.input = input
        self.char2index = {}
        self.index2char = {}
        self.label2index = {}
        self.index2label = {}
        self.maxlen = -1
        
        self.reader = reader(input)
        self.reader.read()
        
        self.model = self._model()

    def _model(self):

        model = Sequential()
        model.add(Embedding(len(self.reader.char2index), 64, input_length=self.reader.maxlen, name='embedding_layer'))
        model.add(LSTM(32, return_sequences=True, name='lstm_layer'))
        model.add(Dense(len(self.reader.label2index), activation='softmax', name='last_layer'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


    def train(self, batch_size=128, nb_epoch=10):
        self.model.fit(self.reader.X_train, self.reader.y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(self.reader.X_test, self.reader.y_test))


    def predict(self, X_test):
        y = self.model.predict_classes(X_test)
        p = self.model.predict_proba(X_test)
        return y, p


    def evaluate(self, batch_size=32):
        return self.model.evaluate(self.reader.X_test, self.reader.y_test, batch_size=batch_size)
        

def main():
    tokenizer = Tokenizer(reader=SequencesIOBReader)
    tokenizer.train(batch_size=128, nb_epoch=10)
    score = tokenizer.evaluate()
    print('Raw test score:', score)

    y_pred, p = tokenizer.predict(tokenizer.reader.X_test)
    #y_test = WindowsIOBWriter().write(tokenizer.reader.y_test)
    y_test, y_pred = SequencesIOBWriter().write(tokenizer.reader.y_test, y_pred)

    # def decode(elems):
    #     # toRet = []
    #     # for sentence in elems:
    #     #     toRet.append([np.where(el==1)[0][0] for el in sentence])
    #     # return toRet
    #     return [np.where(l==1)[0][0] for l in elems]

    # #y_test = list(itertools.chain.from_iterable(decode(tokenizer.reader.y_test)))
    # #y_pred = list(itertools.chain.from_iterable(y_pred))
    # y_test = decode(tokenizer.reader.y_test)
    
    
    print(classification_report(y_test, y_pred, target_names=[tokenizer.reader.index2label[index] for index in sorted(tokenizer.reader.index2label)]))
    print()
    print(confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    main()
