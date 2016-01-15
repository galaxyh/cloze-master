# -*- coding: utf-8 -*-

import codecs
from nltk.tokenize import sent_tokenize, WhitespaceTokenizer


class RCData:
    def __init__(self):
        self._url = ''
        self._sentences = _Text()
        self._question = _Text()
        self._answer = ''
        self._entities = []
        self._num_words = 0
        self._num_sentences = 0

    def load_article(self, filename):
        """Load and parse source article
        :param filename:
        """
        with codecs.open(filename, 'r', encoding='utf8') as f:
            lines = f.readlines()

            self._url = lines[0].strip()

            text = lines[2].strip()
            sentences_raw = sent_tokenize(text)
            sentences_tok = [WhitespaceTokenizer().tokenize(sent) for sent in sentences_raw]
            self._sentences = _Text(sentences_raw, sentences_tok)

            self._num_sentences = len(sentences_raw)
            self._num_words = sum([len(tokens) for tokens in sentences_tok])

            question_raw = lines[4].strip()
            question_tok = WhitespaceTokenizer().tokenize(question_raw)
            self._question = _Text(question_raw, question_tok)

            self._answer = lines[6].strip()

            self._entities = [val.strip() for idx, val in enumerate(lines) if (idx > 7) and val.strip()]

    @property
    def url(self):
        return self._url

    @property
    def sentences(self):
        return self._sentences

    @property
    def question(self):
        return self._question

    @property
    def answer(self):
        return self._answer

    @property
    def entities(self):
        return self._entities

    @property
    def num_words(self):
        return self._num_words

    @property
    def num_sentences(self):
        return self._num_sentences


class _Text:
    def __init__(self, raw='', tokens=[]):
        self._raw = raw
        self._tokens = tokens

    @property
    def raw(self):
        return self._raw

    @raw.setter
    def raw(self, value):
        self._raw = value

    @raw.deleter
    def raw(self):
        del self._raw

    @property
    def tokens(self):
        return self._tokens

    @tokens.setter
    def tokens(self, value):
        self._tokens = value

    @tokens.deleter
    def tokens(self):
        del self._tokens
