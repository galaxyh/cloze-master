# -*- coding: utf-8 -*-
"""Average word embedding trim question entities with heuristic.
"""

import codecs
from optparse import OptionParser

import numpy as np
import scipy.spatial.distance as dist
from mprpc import RPCClient

from rcdata.article import Article


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

    articles = []
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

    for article_filename in nonblank_lines(articles):
        article = Article()
        article.load_article(article_dir + article_filename)

        q_avg = vec_avg(article.question.tokens)
        s_avg = [vec_avg(s) for s in article.sentences.tokens]
        cos = [dist.cosine(q_avg, avg) for avg in s_avg]

        min_dist_1st = float('inf')
        min_idx_1st = 0
        min_dist_2nd = float('inf')
        min_idx_2nd = 0
        for i, (c, s) in enumerate(zip(cos, article.sentences.tokens)):
            if any(t.startswith('@entity') for t in s):
                if c < min_dist_1st:
                    min_dist_2nd = min_dist_1st
                    min_idx_2nd = min_idx_1st
                    min_dist_1st = c
                    min_idx_1st = i

        fmt = '{:<4}{:<1.6f}  {}'
        if options.verbose > 0:
            print 'ARTICLE     >', article_filename
            print 'URL         >', article.url
            print 'QUESTION    >', article.question.raw
            print 'ANSWER      >', article.answer
            print 'GUESS TOP 1 >',
            print fmt.format(min_idx_1st, min_dist_1st, article.sentences.raw[min_idx_1st])
            print 'GUESS TOP 2 >',
            print fmt.format(min_idx_2nd, min_dist_2nd, article.sentences.raw[min_idx_2nd])
            print

        if options.verbose > 1:
            for i, (c, sent) in enumerate(zip(cos, article.sentences.raw)):
                print(fmt.format(i, c, sent))
            print

        q_entities = {e for e in article.question.tokens if e.startswith('@entity')}
        c_entities_1st = {e for e in article.sentences.tokens[min_idx_1st] if e.startswith('@entity')}
        c_entities_2nd = {e for e in article.sentences.tokens[min_idx_2nd] if e.startswith('@entity')}
        c_entities_1st_diff = c_entities_1st - q_entities
        c_entities_2nd_diff = c_entities_2nd - q_entities
        c_entities_intersect = c_entities_1st_diff & c_entities_2nd_diff

        candidates = {}
        if len(c_entities_intersect) > 0:
            candidates = c_entities_intersect
        elif len(c_entities_1st_diff) > 0:
            candidates = c_entities_1st_diff
        else:
            candidates = c_entities_2nd_diff

        fmt = '{},{},{},{}'
        print fmt.format(1 if article.answer in candidates else 0,
                         len(candidates),
                         len(article.entities),
                         article_filename)
