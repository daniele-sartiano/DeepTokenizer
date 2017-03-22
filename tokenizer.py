#!/usr/bin/env python

from __future__ import print_function

import sys
import random
import itertools
import argparse
import numpy as np
import pickle

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LSTM, SimpleRNN, GRU, Dropout, Input, Flatten, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
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

    def save(self):
        return [self.char2index, self.index2char, self.label2index, self.index2label]
    
    def load(self, char2index, index2char, label2index, index2label):
        self.char2index = char2index
        self.index2char = index2char
        self.label2index = label2index
        self.index2label = index2label


    @staticmethod
    def iob2text(input, delimiter='\t'):
        for char_label in input:
            print(char_label, file=sys.stderr)
            char, label = char_label.split(delimiter) if delimiter else char_label
            label = label.strip()
            if label == 'B-S':
                yield '\n%s' % char
            elif label == 'B-T':
                yield ' %s' % char
            elif label == 'I-T':
                yield char

    @staticmethod
    def char2text(input):
        for char in input:
            yield char.split('\n')[0]

    @staticmethod
    def text2iob(input):
        for line in input:
            for i, ch in enumerate(line):
                if ch == '\n':
                    continue
                if ch == ' ':
                    yield ch, 'O'
                elif i == 0 or line[i-1] == '\n':
                    yield ch, 'B-S'
                elif line[i-1] == ' ':
                    yield ch, 'B-T'
                else:
                    yield ch, 'I-T'
                
                

# class SequencesIOBReader(IOBReader):

#     def __init__(self, input, window_size):
#         super(SequencesIOBReader, self).__init__(input)
#         self.window_size = window_size


#     def extractWindows(self, dataset, labels):
#         d = []
#         l = []
#         for i in range(len(dataset)-self.window_size+1):
#             d.append(dataset[i:i+self.window_size])
#             l.append(labels[i:i+self.window_size])
#         return np.asarray(d), np.asarray(l)


#     def extractUniqueWindows(self, dataset):
#         d = []
#         for i in range(0, len(dataset), self.window_size):
#             d.append(dataset[i:i+self.window_size])
#         return d


#     def text2indexes(self, text):
#         chars = []
#         for line in text:
#             for char in line:
#                 chars.append(char)
#         X = self.extractUniqueWindows(chars)
#         X_enc = [[self.char2index[c] for c in x] for x in X]
#         X_enc = pad_sequences(X_enc, maxlen=self.maxlen)
#         return chars, X_enc


#     def read(self, delimiter='\t'):
#         X = []
#         y = []

#         for line in self.input:
#             char, label = (ch for ch in line.split(delimiter))
#             X.append(char)
#             y.append(label.strip())
        
#         for i, c in enumerate(set(X)):
#             self.char2index[c] = i
#             self.index2char[i] = c

#         for i, l in enumerate(set(y)):
#             self.label2index[l] = i
#             self.index2label[i] = l

#         X, y = self.extractWindows(X, y)

#         self.maxlen = max([len(x) for x in X])

#         X_enc = [[self.char2index[c] for c in x] for x in X]
#         self.max_label = max(self.label2index.values()) + 1

#         #one_hot
#         y_enc = [[Reader.encode(self.label2index[c], self.max_label) for c in ey] for ey in y]

#         X_enc = pad_sequences(X_enc, maxlen=self.maxlen)
#         y_enc = pad_sequences(y_enc, maxlen=self.maxlen)

#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_enc, y_enc, random_state=42)


class WindowsIOBReader(IOBReader):

    def __init__(self, input, window_size=None):
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
                v = int(i-step+ii)
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
                v = int(i-step+ii)
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

        X_enc = [[self.char2index[c] if c in self.char2index else self.char2index['<unknown>'] for c in x] for x in X]
        #X_enc = pad_sequences(X_enc, maxlen=self.window_size)
        return chars, X_enc

    
    def read(self, delimiter='\t', dev=True):
        X = []
        y = []

        for line in self.input:
            char, label = (ch for ch in line.split(delimiter))
            X.append(char)
            y.append(label.strip())

        if not self.char2index: # else with load
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
        
        #self.maxlen = max([len(x) for x in X])

        X_enc = [[self.char2index[c] if c in self.char2index else self.char2index['<unknown>'] for c in x] for x in X]

        self.max_label = max(self.label2index.values()) + 1
        
        y_enc = [Reader.encode(self.label2index[c], self.max_label) for c in y]
        #X_enc = pad_sequences(X_enc, maxlen=self.window_size)
        #y_enc = pad_sequences(y_enc)

        if dev:
            X_train, X_dev, y_train, y_dev = train_test_split(X_enc, y_enc, random_state=42)
            return np.asarray(X_train), np.asarray(y_train), np.asarray(X_dev), np.asarray(y_dev)
        else:
            return np.asarray(X_enc), np.asarray(y_enc)


    def save(self):
        fields = super(WindowsIOBReader, self).save()
        fields.append(self.window_size)
        return fields

    
    def load(self, char2index, index2char, label2index, index2label, window_size):
        super(WindowsIOBReader, self).load(char2index, index2char, label2index, index2label)
        self.window_size = window_size



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

    def __init__(self, window_size, max_features, writer, file_model, n_classes, input=sys.stdin, output=sys.stdout):
        self.input = input
        self.window_size = window_size
        self.max_features=max_features
        self.n_classes=n_classes
        self.file_model = file_model
        #self.reader = reader(input, window_size)
        self.writer = writer(output)
        

    def _model(self):
        model = Sequential()

        
        model.add(Embedding(self.max_features, 128, input_length=self.window_size, name='embedding_layer'))

        # # FIXME find a better way for return_sequences
        model.add(LSTM(100, return_sequences=False, name='lstm_layer'))
        model.add(Dense(self.n_classes, activation='softmax', name='last_layer'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


    def train(self, X, y, batch_size=128, nb_epoch=10):
        self.model = self._model()
        self.model.summary()

        self.model.fit(X, 
                       y, 
                       batch_size=batch_size, 
                       nb_epoch=nb_epoch, 
                       validation_split=0.25,
                       #validation_data=(self.reader.X_test, self.reader.y_test),
                       callbacks=[
                           EarlyStopping(verbose=True, patience=5, monitor='val_loss'),
                           ModelCheckpoint('TestModel-progress', monitor='val_loss', verbose=True, save_best_only=True)
                       ])


    def predict(self, X_test, verbose=1):
        y = self.model.predict_classes(X_test, verbose=verbose)
        p = self.model.predict_proba(X_test, verbose=verbose)
        return y, p


    def tokenize(self, text, reader, format='iob'):
        chars, X = reader.text2indexes(text)
        y, p = self.predict(X, verbose=0)
        for i, ch in enumerate(chars):
            yield ch, reader.index2label[y[i]]

        # # sequences
        # index = 0
        # for i in range(0, len(chars), self.window_size):
        #     for ii, ch in enumerate(chars[i:i+self.window_size]):
        #         print(ch, self.reader.index2label[y[index][ii]])
        #     index += 1
        

    def evaluate(self, X_dev, y_dev, batch_size=32):
        return self.model.evaluate(X_dev, y_dev, batch_size=batch_size)
        

    def classification_report(self, y_gold, y_pred, target_names):
        return classification_report(y_gold, y_pred, target_names=target_names)


    def confusion_matrix(self, y_gold, y_pred):
        return confusion_matrix(y_gold, y_pred)


    def save_model(self, reader):
        self.model.save(self.file_model)
        reader_file = open('%s_reader.pickle' % self.file_model, 'wb')
        pickle.dump(reader.save(), reader_file)
        reader_file.close()


    def load(self, reader):
        self.model = load_model(self.file_model)
        reader_file = open('%s_reader.pickle' % self.file_model, 'rb')
        reader.load(*pickle.load(reader_file))



def main():
    parser = argparse.ArgumentParser(description='DeepTokenizer')
    subparsers = parser.add_subparsers()

    common_args = [
        (['-w', '--window-size'], {'help':'window size', 'type':int, 'default':5}),
        (['-f', '--file-model'], {'help':'file model', 'type':str, 'default':'tokenizer.model'}),
    ]

    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(which='train')
    parser_train.add_argument('-epochs', '--epochs', help='Epochs', type=int, default=20)
    parser_train.add_argument('-batch', '--batch', help='# batch', type=int, default=256)
    for arg in common_args:
        parser_train.add_argument(*arg[0], **arg[1])
    
    parser_predict = subparsers.add_parser('predict')
    parser_predict.set_defaults(which='predict')
    for arg in common_args:
        parser_predict.add_argument(*arg[0], **arg[1])

    parser_test = subparsers.add_parser('test')
    parser_test.set_defaults(which='test')
    for arg in common_args:
        parser_test.add_argument(*arg[0], **arg[1])
    
    parser_iob2text = subparsers.add_parser('iob2text')
    parser_iob2text.set_defaults(which='iob2text')

    parser_char2text = subparsers.add_parser('char2text')
    parser_char2text.set_defaults(which='char2text')

    parser_text2iob = subparsers.add_parser('text2iob')
    parser_text2iob.set_defaults(which='text2iob')


    args = parser.parse_args()

    if args.which == 'train':
        #tokenizer = Tokenizer(window_size=args.window_size, reader=SequencesIOBReader, writer=SequencesIOBWriter)
        
        reader = WindowsIOBReader(input=sys.stdin, window_size=args.window_size)
        X_train, y_train, X_dev, y_dev = reader.read()

        tokenizer = Tokenizer(window_size=args.window_size, max_features=len(reader.char2index), n_classes=len(reader.label2index), file_model=args.file_model, writer=WindowsIOBWriter)
        
        tokenizer.train(X_train, y_train, batch_size=args.batch, nb_epoch=args.epochs)
        tokenizer.save_model(reader)
        score = tokenizer.evaluate(X_dev, y_dev)
        print('Raw test score:', score)

        y_pred, p = tokenizer.predict(X_dev)

        print(tokenizer.classification_report(tokenizer.writer.write(y_dev), y_pred, [reader.index2label[index] for index in sorted(reader.index2label)]))
        print()
        print(tokenizer.confusion_matrix(tokenizer.writer.write(y_dev), y_pred))

    elif args.which == 'predict':
        #tokenizer = Tokenizer(window_size=args.window_size, reader=SequencesIOBReader, writer=SequencesIOBWriter)
        tokenizer = Tokenizer(window_size=args.window_size, max_features=None, n_classes=None, file_model=args.file_model, writer=WindowsIOBWriter)
        reader = WindowsIOBReader(input=sys.stdin)
        tokenizer.load(reader)
        #y, p = tokenizer.tokenize("Domani vado in ufficio.Dopodomani vado al mare!Ho voglia di vedere un \"bel\" film.")
        for token in IOBReader.iob2text(tokenizer.tokenize(sys.stdin.read(), reader), delimiter=None):
            print(token, end='')

    elif args.which == 'test':        
        tokenizer = Tokenizer(window_size=args.window_size, max_features=None, n_classes=None, file_model=args.file_model, writer=WindowsIOBWriter)
        reader = WindowsIOBReader(input=sys.stdin)
        tokenizer.load(reader)
        X, y = reader.read(dev=False)

        score = tokenizer.evaluate(X, y)

        y_pred, p = tokenizer.predict(X)

        print(tokenizer.classification_report(tokenizer.writer.write(y), y_pred, [reader.index2label[index] for index in sorted(reader.index2label)]))
        print()
        print(tokenizer.confusion_matrix(tokenizer.writer.write(y), y_pred))

    elif args.which == 'iob2text':
        IOBReader.iob2text(sys.stdin)

    elif args.which == 'char2text':
        for ch in IOBReader.char2text(sys.stdin):
            print(ch, end='')

    elif args.which == 'text2iob':
        for ch, iob in IOBReader.text2iob(sys.stdin):
            print('%s\t%s' % (ch, iob))

        
if __name__ == '__main__':
    main()
    import gc; gc.collect()
