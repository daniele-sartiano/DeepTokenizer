#!/usr/bin/env python3

import sys
import argparse


class Converter(object):
    def __init__(self, input=sys.stdin, output=sys.stdout):
        self.input = input
        self.output = output

    def run(self):
        index = 0
        skip = set()
        for line in self.input:
            line = line.strip()
            if line.startswith('#'):
                if 'text' in line:
                    original = line.split('=', 1)[1].strip()
                    index = -1
                continue
            elif not line:
                continue
            
            ids, token = line.split('\t')
            
            if '.' in ids:
                continue

            ids = [int(el) for el in ids.split('-')]
            if len(ids) > 1:
                skip = set(range(ids[0], ids[1]+1))
            elif ids[0] in skip:
                skip.discard(ids[0])
                continue

            for index_token, ch in enumerate(token):
                index +=1
                iob = ''
                if index == 0:
                    iob = 'B-S'
                elif index_token == 0:
                    iob = 'B-T'
                else:
                    iob = 'I-T'
                print('%s\t%s' % (ch, iob))
                            
            while len(original) > index+1 and original[index+1] == ' ':
                index += 1
                print(' \tO')
                

def main():
    parser = argparse.ArgumentParser(description='Build a trainset for a Tokenizer starting from UD')
    converter = Converter()
    converter.run()


if __name__ == '__main__':
    main()
