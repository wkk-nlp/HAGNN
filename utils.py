# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import torch.nn as nn
from nltk.corpus import stopwords
import nltk.tokenize
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME,CONFIG_NAME,BertPreTrainedModel,BertModel

MAX_VOCAB_SIZE = 10000
PAD, CLS, UNK = '[PAD]', '[CLS]', '[UNK]' 
# stop_words = stopwords.words('english')
stop_words = []
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'\'', '\'', '`', '``',
                '-', '--', '|', '\/']
stop_words.extend(punctuations)


def build_vocab(max_size, min_freq):
    vocab_dic = {}
    with open("./data/MAMS/overall.txt", 'r', encoding='gbk') as f:
        for lin in tqdm(f):
            if not lin:
                continue
            content = lin.split('\t')[0]
            for i in punctuations:
                content = content.replace(i, '')
            content = content.split(' ')
            for word in content:
                if word in stop_words:
                    continue
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq],
                            key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    print(len(vocab_dic))
    return vocab_dic


def build_dataset(dataset, ues_word):
    if os.path.exists('data/' + dataset + '/vocab.pkl'):
        vocab = pkl.load(open('data/' + dataset + '/vocab.pkl', 'rb'))
    else:
        vocab = build_vocab(max_size=MAX_VOCAB_SIZE, min_freq=3)
        pkl.dump(vocab, open('data/' + dataset + '/vocab.pkl', 'wb'))
    print(f"Vocab size: {len(vocab)}")
    word_tokenizer = lambda x: [y for y in x]
    tokenizer = BertTokenizer.from_pretrained('./bert_base_uncased')
    def load_dataset(path, pad_size=8):
        contents = []
        label_list = []
        masks = []
        words = []
        with open(path, 'r', encoding='gbk') as f:
            for line in f.readlines():
                words_line = []
                lin = line.strip()
                if not lin:
                    continue
                content, labels = lin.split('\t')
                token = tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = tokenizer.convert_tokens_to_ids(token)
                label_list.append(int(labels))
                word_token = content.replace(',', '').replace('.', '').split(' ')
                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                    if len(word_token) < pad_size:
                        word_token.extend([PAD] * (pad_size - len(word_token)))
                    else:
                        word_token = word_token[:pad_size]
                for word in word_tokenizer(word_token):
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                words.append(words_line)
                contents.append(token_ids)
                masks.append(mask)
        return vocab, contents, masks, label_list, words
    vocab, train, mask, label, words = load_dataset('data/' + dataset + '/overall.txt', 16)
    print('loading data')
    return vocab, train, mask, label, words


class DatasetIterater(object):
    def __init__(self, batches, labels, batch_size):
        self.batch_size = batch_size
        self.batches = batches
        self.labels = labels
        self.n_batches = len(batches) // batch_size
        self.residue = False 
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0

    @staticmethod
    def _to_tensor(datas, labels):
        x = [data for data in datas]
        y = [label for label in labels]

        return x, y

    def __next__(self):
        if self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            labels = self.labels[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches, labels)
            return batches, labels

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches