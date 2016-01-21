# -*- coding: utf-8 -*-
"""rc-data statistics
"""

import codecs
import multiprocessing
from optparse import OptionParser

from article import Article


def parse_args():
    p = OptionParser()
    p.add_option('-v', '--verbose', action='store',
                 dest='verbose', type='int', default='0')
    p.add_option('-a', '--article', action='store',
                 dest='article', type='string', default='')
    p.add_option('-l', '--article_list', action='store',
                 dest='article_list', type='string', default='')
    p.add_option('-d', '--article_dir', action='store',
                 dest='article_dir', type='string', default='')
    return p.parse_args()


def nonblank_lines(source):
    for l in source:
        line = l.strip()
        if line:
            yield line


def mp_worker(file_name):
    rcd = Article()
    rcd.load_article(file_name)
    return rcd.num_words, rcd.num_sentences, len(rcd.entities)

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

    num_cpu = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cpu)

    num_articles = 0
    num_words = 0
    num_sentences = 0
    num_entities = 0

    for w, s, e in pool.imap(mp_worker, [article_dir + a for a in nonblank_lines(articles)]):
        num_articles += 1
        num_words += w
        num_sentences += s
        num_entities += e

    print 'Number of articles:', num_articles
    print 'Average number of words per article:    ', float(num_words) / num_articles
    print 'Average number of sentences per article:', float(num_sentences) / num_articles
    print 'Average number of entities per article: ', float(num_entities) / num_articles
