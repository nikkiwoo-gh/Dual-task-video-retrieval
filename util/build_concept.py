# Create a concept list from captions
# Jiaxin Wu
# 2020.02.04

from __future__ import print_function
import pickle
from collections import Counter
import json
import argparse
import os
import sys
import re

from basic.constant import ROOT_PATH, logger
from basic.common import makedirsforfile, checkToSkip
from basic.generic_utils import Progbar



class Concept(object):
    """ concept wrapper"""

    def __init__(self, text_style):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.text_style = text_style

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx and 'bow' not in self.text_style:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def from_flickr_json(path):
    dataset = json.load(open(path, 'r'))['images']
    captions = []
    for i, d in enumerate(dataset):
        captions += [str(x['raw']) for x in d['sentences']]

    return captions

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    return string.strip().lower().split()

def from_txt(txt):
    captions = []
    with open(txt, 'rb') as reader:
        for line in reader:
            line = line.decode()
            cap_id, caption = line.split(' ', 1)
            captions.append(caption.strip())
    return captions

def build_concept(collection, text_style, threshold=4, rootpath=ROOT_PATH):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    cap_file = os.path.join(rootpath, collection, 'TextData', '%s.caption.txt'%collection)
    captions = from_txt(cap_file)
    pbar = Progbar(len(captions))

    stop_word_file = os.path.join(ROOT_PATH, 'stopwords_en.txt')
    stop_words = []
    with open(stop_word_file, 'rb') as reader:
        for word in reader:
            word = word.decode().strip()
            stop_words.append(word)

    for i, caption in enumerate(captions):
        tokens = clean_str(caption.lower())
        counter.update(tokens)

        pbar.add(1)
        # if i % 1000 == 0:
        #     print("[%d/%d] tokenized the captions." % (i, len(captions)))

    # Discard if the occurrence of the word is less than min_word_cnt.
    words = []
    for word, cnt in counter.items():
        if cnt >= threshold:
            if not word in stop_words:
                words.append(word)

    # Create a vocab wrapper and add some special tokens.
    vocab = Concept(text_style)

    # Add words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab, counter


def main(option):
    rootpath = option.rootpath
    collection = option.collection
    threshold = option.threshold
    text_style = option.text_style

    vocab_file = os.path.join(rootpath, collection, 'TextData', 'concept', 'concept_list_gt%d.pkl'%threshold)
    counter_file = os.path.join(os.path.dirname(vocab_file), 'concept_frequency_count_gt%s.txt'%threshold)

    if checkToSkip(vocab_file, option.overwrite):
        sys.exit(0)
    makedirsforfile(vocab_file)

    stop_word_file = os.path.join(ROOT_PATH, 'stopwords_en.txt')
    stop_words = []
    with open(stop_word_file, 'rb') as reader:
        for word in reader:
            word = word.decode().strip()
            stop_words.append(word)

    concept, concept_counter = build_concept(collection, text_style, threshold=threshold, rootpath=rootpath)
    with open(vocab_file, 'wb') as writer:
        pickle.dump(concept, writer, pickle.HIGHEST_PROTOCOL)
    logger.info("Saved vocabulary file to %s", vocab_file)

    concept_counter_list = []
    for word, cnt in concept_counter.items():
        if cnt >= threshold:
            if not word in stop_words:
                concept_counter_list.append([word, cnt])
    concept_counter_list.sort(key=lambda x: x[1], reverse=True)
    with open(counter_file, 'w') as writer:
        writer.write('\n'.join(map(lambda x: x[0]+' %d'%x[1], concept_counter_list)))
    logger.info("Saved vocabulary counter file to %s", counter_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH, help='root path. (default: %s)'%ROOT_PATH)
    parser.add_argument('collection', type=str, help='collection tgif|msrvtt10k')
    parser.add_argument('--threshold', type=int, default=5, help='threshold to build vocabulary. (default: 5)')
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1], help='overwrite existed vocabulary file. (default: 0)')
    parser.add_argument('--text_style', type=str, choices=['rnn', 'bow'], default='bow',
                        help='text style for vocabulary. (default: bow)')
    opt = parser.parse_args()
    print(json.dumps(vars(opt), indent = 2))

    main(opt)

