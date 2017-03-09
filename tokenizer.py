#!/usr/bin/env python
from __future__ import print_function

import sys
import random
import itertools
import argparse
import numpy as np
import cPickle as pickle

from keras.models import Sequential, load_model
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
        self.maxlen = -1

        self.X_train = []
        self.X_test = []

        self.y_train = []
        self.y_test = []
    
    def save(self):
        return [self.char2index, self.index2char, self.label2index, self.index2label, self.maxlen]
    
    def load(self, char2index, index2char, label2index, index2label, maxlen):
        self.char2index = char2index
        self.index2char = index2char
        self.label2index = label2index
        self.index2label = index2label
        self.maxlen = maxlen


class SequencesIOBReader(IOBReader):

    def __init__(self, input, window_size):
        super(SequencesIOBReader, self).__init__(input)
        self.window_size = window_size


    def extractWindows(self, dataset, labels):
        d = []
        l = []
        for i in range(len(dataset)-self.window_size+1):
            d.append(dataset[i:i+self.window_size])
            l.append(labels[i:i+self.window_size])
        return np.asarray(d), np.asarray(l)


    def extractUniqueWindows(self, dataset):
        d = []
        for i in range(0, len(dataset), self.window_size):
            d.append(dataset[i:i+self.window_size])
        return d


    def text2indexes(self, text):
        chars = []
        for line in text:
            for char in line:
                chars.append(char)
        X = self.extractUniqueWindows(chars)
        X_enc = [[self.char2index[c] for c in x] for x in X]
        X_enc = pad_sequences(X_enc, maxlen=self.maxlen)
        return chars, X_enc


    def read(self, delimiter='\t'):
        X = []
        y = []

        for line in self.input:
            char, label = (ch for ch in line.split(delimiter))
            X.append(char)
            y.append(label.strip())
        
        for i, c in enumerate(set(X)):
            self.char2index[c] = i
            self.index2char[i] = c

        for i, l in enumerate(set(y)):
            self.label2index[l] = i
            self.index2label[i] = l

        X, y = self.extractWindows(X, y)

        self.maxlen = max([len(x) for x in X])

        X_enc = [[self.char2index[c] for c in x] for x in X]
        self.max_label = max(self.label2index.values()) + 1

        #one_hot
        y_enc = [[Reader.encode(self.label2index[c], self.max_label) for c in ey] for ey in y]

        X_enc = pad_sequences(X_enc, maxlen=self.maxlen)
        y_enc = pad_sequences(y_enc, maxlen=self.maxlen)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_enc, y_enc, random_state=42)


class WindowsIOBReader(IOBReader):

    def __init__(self, input, window_size):
        super(WindowsIOBReader, self).__init__(input)
        self.window_size = window_size


    def extractWindows(self, dataset, labels):
        d = []
        l = []

        step = (self.window_size-1)/2
        for i, label in enumerate(labels):
            l.append(label)
            example = []
            
            for ii in range(self.window_size):
                v = i-step+ii
                if v < 0:
                    example.append('<padding>')
                elif v >= len(dataset):
                    example.append('<padding>')
                else:
                    example.append(dataset[v])
            d.append(example)

        return np.asarray(d), np.asarray(l)


    def extractUniqueWindows(self, dataset):
        d = []
        
        step = (self.window_size-1)/2
        
        for i in range(len(dataset)):
            example = []
            
            for ii in range(self.window_size):
                v = i-step+ii
                if v < 0:
                    example.append('<padding>')
                elif v >= len(dataset):
                    example.append('<padding>')
                else:
                    example.append(dataset[v])
            d.append(example)
            
            
        # for i in range(len(dataset)-self.window_size+1):
        #     d.append(dataset[i:i+self.window_size])
        return np.asarray(d)


    def text2indexes(self, text):
        chars = []
        for line in text:
            for char in line:
                chars.append(char)
        X = self.extractUniqueWindows(chars)
        print(X)

        #X_enc = [[self.char2index[c] for c in x] for x in X]

        X_enc =  []
        for x in X:
            w = []
            for c in x:
                try:
                    w.append(self.char2index[c])
                except:
                    print('%s unknown' % c, file=sys.stderr)
                    w.append(self.char2index['<unknown>'])
            X_enc.append(w)

        X_enc = pad_sequences(X_enc, maxlen=self.maxlen)

        return chars, X_enc

    
    def read(self, delimiter='\t'):
        X = []
        y = []

        for line in self.input:
            char, label = (ch for ch in line.split(delimiter))
            X.append(char)
            y.append(label.strip())
        
        for i, c in enumerate(set(X)):
            self.char2index[c] = i
            self.index2char[i] = c
        self.index2char[len(self.index2char)] = '<padding>'
        self.char2index['<padding>'] = len(self.index2char)-1
        self.index2char[len(self.index2char)] = '<unknown>'
        self.char2index['<unknown>'] = len(self.index2char)-1

        for i, l in enumerate(set(y)):
            self.label2index[l] = i
            self.index2label[i] = l

        X, y = self.extractWindows(X, y)

        self.maxlen = max([len(x) for x in X])

        X_enc = [[self.char2index[c] for c in x] for x in X]

        self.max_label = max(self.label2index.values()) + 1

        y_enc = [Reader.encode(self.label2index[c], self.max_label) for c in y]
        X_enc = pad_sequences(X_enc, maxlen=self.maxlen)
        y_enc = pad_sequences(y_enc)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_enc, y_enc, random_state=42)


class Writer(object):

    def __init__(self, output):
        self.output = output

        
class WindowsIOBWriter(Writer):

    @staticmethod
    def decode(elems):
        return [np.where(l==1)[0][0] for l in elems]


    def write(self, y_test):
        return WindowsIOBWriter.decode(y_test)


class SequencesIOBWriter(Writer):
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



class Tokenizer(object):

    def __init__(self, window_size=13, reader=WindowsIOBReader, writer=WindowsIOBWriter, file_model='tokenizer.model', input=sys.stdin, output=sys.stdout):
        self.input = input
        self.window_size = window_size
        self.char2index = {}
        self.index2char = {}
        self.label2index = {}
        self.index2label = {}
        
        self.maxlen = -1
        
        self.file_model = file_model
        self.reader = reader(input, window_size)
        self.writer = writer(output)
        

    def _model(self):
        model = Sequential()
        model.add(Embedding(len(self.reader.char2index), 64, input_length=self.reader.maxlen, name='embedding_layer'))

        # FIXME return_sequences=self.reader==SequencesIOBReader <-- find a better way
        model.add(LSTM(32, return_sequences=self.reader==SequencesIOBReader, name='lstm_layer'))
        model.add(Dense(len(self.reader.label2index), activation='softmax', name='last_layer'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


    def train(self, batch_size=128, nb_epoch=10):
        self.reader.read()
        self.model = self._model()
        
        self.model.fit(self.reader.X_train, self.reader.y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(self.reader.X_test, self.reader.y_test))


    def predict(self, X_test):
        y = self.model.predict_classes(X_test)
        p = self.model.predict_proba(X_test)
        return y, p


    def tokenize(self, text, format='iob'):
        chars, X = self.reader.text2indexes(text)
        y, p = self.predict(X)

        print(len(chars))
        print(len(y))

        for i, ch in enumerate(chars):
            print(ch, self.reader.index2label[y[i]])

        # # sequences
        # index = 0
        # for i in range(0, len(chars), self.window_size):
        #     for ii, ch in enumerate(chars[i:i+self.window_size]):
        #         print(ch, self.reader.index2label[y[index][ii]])
        #     index += 1

        return y, p
        

    def evaluate(self, batch_size=32):
        return self.model.evaluate(self.reader.X_test, self.reader.y_test, batch_size=batch_size)
        

    def classification_report(self, y_pred):
        #y_test, y_pred = self.writer.write(self.reader.y_test, y_pred)
        y_test = self.writer.write(self.reader.y_test)
        return classification_report(y_test, y_pred, target_names=[self.reader.index2label[index] for index in sorted(self.reader.index2label)])


    def confusion_matrix(self, y_pred):
        #y_test, y_pred = self.writer.write(self.reader.y_test, y_pred)
        y_test = self.writer.write(self.reader.y_test)
        return confusion_matrix(y_test, y_pred)


    def save_model(self):
        self.model.save(self.file_model)
        reader_file = open('%s_reader.pickle' % self.file_model, 'wb')
        pickle.dump(self.reader.save(), reader_file)
        reader_file.close()


    def load(self):
        self.model = load_model(self.file_model)
        reader_file = open('%s_reader.pickle' % self.file_model, 'rb')
        self.reader.load(*pickle.load(reader_file))



def main():
    parser = argparse.ArgumentParser(description='DeepTokenizer')
    subparsers = parser.add_subparsers()

    common_args = [
        (['-w', '--window-size'], {'help':'window size', 'type':int, 'default':5})
    ]

    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(which='train')
    parser_train.add_argument('-epochs', '--epochs', help='Epochs', type=int, default=20)
    parser_train.add_argument('-batch', '--batch', help='# batch', type=int, default=128)
    for arg in common_args:
        parser_train.add_argument(*arg[0], **arg[1])
    
    parser_predict = subparsers.add_parser('predict')
    parser_predict.set_defaults(which='predict')
    for arg in common_args:
        parser_predict.add_argument(*arg[0], **arg[1])
    

    args = parser.parse_args()

    if args.which == 'train':
        #tokenizer = Tokenizer(window_size=args.window_size, reader=SequencesIOBReader, writer=SequencesIOBWriter)
        tokenizer = Tokenizer(window_size=args.window_size, reader=WindowsIOBReader, writer=WindowsIOBWriter)
        tokenizer.train(batch_size=args.batch, nb_epoch=args.epochs)
        tokenizer.save_model()
        score = tokenizer.evaluate()
        print('Raw test score:', score)

        y_pred, p = tokenizer.predict(tokenizer.reader.X_test)

        print(tokenizer.classification_report(y_pred))
        print()
        print(tokenizer.confusion_matrix(y_pred))

        y, p = tokenizer.tokenize('domani vado al mare.Dopodomani no.')

    elif args.which == 'predict':
        #tokenizer = Tokenizer(window_size=args.window_size, reader=SequencesIOBReader, writer=SequencesIOBWriter)
        tokenizer = Tokenizer(window_size=args.window_size, reader=WindowsIOBReader, writer=WindowsIOBWriter)
        tokenizer.load()
        y, p = tokenizer.tokenize("Domani vado in ufficio.Dopodomani vado al mare!Ho voglia di vedere un \"bel\" film.")
    
if __name__ == '__main__':
    main()
