# -*- coding: utf-8 -*-

import codecs
import logging
import time

import numpy as np
from keras.layers.core import Activation
from seq2seq.models import AttentionSeq2seq

import w2v

logging.basicConfig(filename=(__name__ + '.log'), level=logging.INFO)
_logger = logging.getLogger(__name__)

EOS_SYMBOL = '$$$'
UKN_TOKEN = '###'


def train_test(train_pair, test_pair, nb_epoch, input_dim, input_length, hidden_dim, output_length, output_dim, depth):
    (x_train, y_train) = train_pair
    (x_test, y_test) = test_pair

    _logger.info('x_train shape: {}'.format(x_train.shape))
    _logger.info('y_train shape: {}'.format(y_train.shape))
    _logger.info('x_test shape: {}'.format(x_test.shape))
    _logger.info('y_test shape: {}'.format(y_test.shape))

    ts0 = time.time()
    ts1 = time.time()

    _logger.info('Building model...')
    model = AttentionSeq2seq(input_dim=input_dim,
                             input_length=input_length,
                             hidden_dim=hidden_dim,
                             output_length=output_length,
                             output_dim=output_dim,
                             depth=depth)
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    _logger.info('Done building model ({:.1f} minutes).'.format((time.time() - ts1) / 60))

    ts1 = time.time()

    _logger.info('Training...')
    model.fit(x_train, y_train, nb_epoch=nb_epoch, show_accuracy=True)
    _logger.info('Done training ({:.1f} minutes).'.format((time.time() - ts1) / 60))

    ts1 = time.time()

    _logger.info('Evaluating...')
    objective_score = model.evaluate(x_test, y_test)
    _logger.info('Objective score = {}'.format(objective_score))
    _logger.info('Done evaluation ({:.1f} minutes)'.format((time.time() - ts1) / 60))

    _logger.info('Predicted:\n{}\nGround truth:\n{}'.format(model.predict(x_test), y_test))
    _logger.info('Total time elapsed: {:.1f} minutes.'.format((time.time() - ts0) / 60))


def word_one_hot(w2vm, word):
    v = np.zeros(len(w2vm.vocab))
    v[w2vm.vocab[word].index] = 1.0
    return v


def sentence_one_hot(maxlen, w2vm, sentence):
    s = np.full((maxlen, len(w2v_model.vocab)), word_one_hot(w2vm, EOS_SYMBOL))
    for i, t in enumerate(sentence):
        s[i] = word_one_hot(w2vm, t)
    return s


if __name__ == '__main__':
    corpus_en_filename = 'corpus.en.txt'
    corpus_ch_filename = 'corpus.ch.txt'

    sentences_en = []
    sentences_ch = []
    for line in codecs.open(corpus_en_filename, 'r', 'utf-8'):
        tokens = line.split()
        tokens.append(EOS_SYMBOL)
        sentences_en.append(tokens)
    maxlen_en = len(max(sentences_en, key=len))

    for line in codecs.open(corpus_ch_filename, 'r', 'utf-8'):
        tokens = line.split()
        tokens.append(EOS_SYMBOL)
        sentences_ch.append(tokens)
    maxlen_ch = len(max(sentences_ch, key=len))

    sentences_all = sentences_en + sentences_ch

    w2v_param = {'win_size': 3,
                 'min_w_num': 1,
                 'vect_size': 10,
                 'workers_num': 1,
                 'corpus_name': 'corpus.txt',
                 'save_path': '.',
                 'new_models_dir': 'w2v_model'}

    w2v_model = w2v.get_model(w2v_param, sentences_all)

    x_train = np.empty((len(sentences_en), maxlen_en, len(w2v_model.vocab)))
    for i, s in enumerate(sentences_en):
        x_train[i] = sentence_one_hot(maxlen_en, w2v_model, s)

    y_train = np.empty((len(sentences_ch), maxlen_ch, len(w2v_model.vocab)))
    for i, s in enumerate(sentences_ch):
        y_train[i] = sentence_one_hot(maxlen_ch, w2v_model, s)

    '''x_train = np.asarray([
        [[0, 1], [0, 1], [0, 1]],
        [[0, 1], [0, 1], [1, 0]],
        [[0, 1], [1, 0], [0, 1]],
        [[1, 0], [0, 1], [1, 0]]
    ])

    y_train = np.asarray([
        [[0, 1], [1, 0], [1, 0]],
        [[0, 1], [1, 0], [0, 1]],
        [[1, 0], [0, 1], [0, 1]],
        [[1, 0], [1, 0], [0, 1]]
    ])'''

    x_test = x_train.copy()
    y_test = y_train.copy()

    x_train = np.vstack([x_train, x_train])
    y_train = np.vstack([y_train, y_train])

    input_dim = x_train.shape[2]
    input_length = x_train.shape[1]
    hidden_dim = 100
    output_length = y_train.shape[1]
    output_dim = y_train.shape[2]
    depth = 4

    train_test((x_train, y_train), (x_test, y_test), 1000, input_dim, input_length, hidden_dim, output_length,
               output_dim, depth)
