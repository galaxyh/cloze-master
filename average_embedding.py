# -*- coding: utf-8 -*-

import codecs
import numpy as np
import scipy.spatial.distance as dist
from mprpc import RPCClient
from optparse import OptionParser
from rcdata import RCData


def parse_args():
    p = OptionParser()
    p.add_option('-s', '--server_ip', action='store',
                 dest='server_ip', type='string', default='127.0.0.1')
    p.add_option('-p', '--server_port', action='store',
                 dest='server_port', type='int', default='1979')
    p.add_option('-d', '--embedding_dim', action='store',
                 dest='embedding_dim', type='int', default='300')
    p.add_option('-v', '--verbose', action='store',
                 dest='verbose', type='int', default='0')
    p.add_option('-a', '--article', action='store',
                 dest='article', type='string', default='')
    p.add_option('-l', '--article_list', action='store',
                 dest='article_list', type='string', default='')
    p.add_option('-r', '--article_dir', action='store',
                 dest='article_dir', type='string', default='')
    return p.parse_args()


def nonblank_lines(source):
    for l in source:
        line = l.strip()
        if line:
            yield line


def vec_avg(sentence):
    """Calculate averaged word embedding of the sentence

    :param sentence:
    :return: Averaged word embedding vector
    """
    unk = 0
    vec_sum = np.full(options.embedding_dim, 0.00000001)
    for w in sentence:
        if client.call('exist', w):
            vec_sum = np.add(vec_sum, client.call('vector', w))
        else:
            unk += 1

    if unk < len(sentence):
        return np.divide(vec_sum, len(sentence) - unk)
    else:
        return vec_sum


if __name__ == '__main__':
    options, remainder = parse_args()

    if options.article_list.strip():
        articles = codecs.open(options.article_list.strip(), 'r', encoding='utf8')
    elif options.article.strip():
        articles = [options.article.strip()]
    else:
        quit()

    article_dir = options.article_dir.strip()
    if article_dir and not article_dir.endswith('/'):
        article_dir += '/'

    client = RPCClient(options.server_ip, options.server_port)

    for article in nonblank_lines(articles):
        data = RCData()
        data.load_article(article_dir + article)

        q_avg = vec_avg(data.question.tokens)
        s_avg = [vec_avg(s) for s in data.sentences.tokens]
        cos = [dist.cosine(q_avg, avg) for avg in s_avg]

        min_dist = float('inf')
        min_idx = 0
        for i, (c, s) in enumerate(zip(cos, data.sentences.tokens)):
            if any(t.startswith('@entity') for t in s):
                if c < min_dist:
                    min_dist = c
                    min_idx = i

        fmt = '{:<4}{:<1.6f}  {}'
        if options.verbose > 0:
            print 'ARTICLE  >', article
            print 'URL      >', data.url
            print 'QUESTION >', data.question.raw
            print 'ANSWER   >', data.answer
            print 'GUESS    >',
            print fmt.format(min_idx, min_dist, data.sentences.raw[min_idx])
            print

        if options.verbose > 1:
            for i, (c, sent) in enumerate(zip(cos, data.sentences.raw)):
                print(fmt.format(i, c, sent))
            print

        fmt = '{},{},{},{}'
        print fmt.format(1 if data.answer in data.sentences.tokens[min_idx] else 0,
                         len([e for e in data.sentences.tokens[min_idx] if e.startswith('@entity')]),
                         len(data.entities),
                         article)
