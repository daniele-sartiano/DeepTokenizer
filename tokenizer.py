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


class WindowsIOBReader(IOBReader):

    @staticmethod
    def extractWindows(dataset, labels, window_size):
        d = []
        l = []
        for i in range(len(dataset)-window_size+1):
            d.append(dataset[i:i+window_size])
            #l.append(labels[window_size+i-3])
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

        X, y = WindowsIOBReader.extractWindows(X, y, 5)

        self.maxlen = max([len(x) for x in X])

        X_enc = [[self.char2index[c] for c in x] for x in X]
        self.max_label = max(self.label2index.values()) + 1

        print(y[0])

        #y_enc = [[self.label2index[c] for c in ey] for ey in y]
        #one_hot
        y_enc = [[Reader.encode(self.label2index[c], self.max_label) for c in ey] for ey in y]

        # y_enc = [Reader.encode(self.label2index[c], self.max_label) for c in y]
        # X_enc = pad_sequences(X_enc, maxlen=self.maxlen)
        # y_enc = pad_sequences(y_enc, maxlen=self.maxlen)

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
        model.add(Embedding(len(self.reader.char2index), 128, input_length=self.reader.maxlen))

        model.add(LSTM(32, return_sequences=True))
        model.add(Dense(len(self.reader.label2index), activation='softmax'))

        #model.add(TimeDistributed(len(self.label2index)))
        #model.add(Dense(len(self.label2index)))
        #model.add(Activation('softmax'))
        #model.add(Dropout(0.5))

        #model.add(Dense(len(self.label2index), activation='softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # model.add(LSTM(64, init='glorot_uniform', inner_init='orthogonal',
        #                activation='tanh', inner_activation='hard_sigmoid', 
        #                return_sequences=False))
        # model.add(Dropout(0.5))
        # model.add(Dense(len(self.label2index)))
        # model.add(Activation('softmax'))
        
        # model.compile(loss='categorical_crossentropy', optimizer='adam', class_mode='categorical')

        return model


    def train(self, batch_size=128, nb_epoch=10):
        self.model.fit(self.reader.X_train, self.reader.y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(self.reader.X_test, self.reader.y_test))


    def predict(self, X_test):
        y = self.model.predict_classes(X_test)
        p = self.model.predict_proba(X_test)
        return y, p


    def evaluate(self, batch_size=32):
        return self.model.evaluate(self.reader.X_test, self.reader.y_test, batch_size=batch_size)


    # def score(yh, pr):
    #     coords = [np.where(yhh > 0)[0][0] for yhh in yh]
    #     yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    #     ypr = [prr[co:] for prr, co in zip(pr, coords)]
    #     fyh = [c for row in yh for c in row]
    #     fpr = [c for row in ypr for c in row]
    #     return fyh, fpr
        

def main():
    tokenizer = Tokenizer()
    #X_train, X_test, y_train, y_test = tokenizer.read()

    tokenizer.train(batch_size=128, nb_epoch=10)
    score = tokenizer.evaluate()
    print('Raw test score:', score)

    y_pred, p = tokenizer.predict(tokenizer.reader.X_test)

    def decode(elems):
        toRet = []
        for sentence in elems:
            toRet.append([np.where(el==1)[0][0] for el in sentence])
        return toRet

    y_test = list(itertools.chain.from_iterable(decode(tokenizer.reader.y_test)))
    y_pred = list(itertools.chain.from_iterable(y_pred))
    
    print(classification_report(y_test, y_pred, target_names=[tokenizer.reader.index2label[index] for index in sorted(tokenizer.reader.index2label)]))
    print()
    print(confusion_matrix(y_test, y_pred))



# pr = model.predict_classes(X_train)
# yh = y_train.argmax(2)
# fyh, fpr = score(yh, pr)
# print 'Training accuracy:', accuracy_score(fyh, fpr)
# print 'Training confusion matrix:'
# print confusion_matrix(fyh, fpr)
# precision_recall_fscore_support(fyh, fpr)

# pr = model.predict_classes(X_test)
# yh = y_test.argmax(2)
# fyh, fpr = score(yh, pr)
# print 'Testing accuracy:', accuracy_score(fyh, fpr)
# print 'Testing confusion matrix:'
# print confusion_matrix(fyh, fpr)
# precision_recall_fscore_support(fyh, fpr)

if __name__ == '__main__':
    main()
