import numpy as np
import scipy.spatial.distance as dist
from mprpc import RPCClient
from nltk.tokenize import sent_tokenize, WhitespaceTokenizer


def parse_args():
    from optparse import OptionParser
    p = OptionParser()
    p.add_option('-s', '--server_ip', action='store',
                 dest='server_ip', type='string', default='127.0.0.1')
    p.add_option('-p', '--server_port', action='store',
                 dest='server_port', type='int', default='1979')
    p.add_option('-d', '--embedding_dim', action='store',
                 dest='embedding_dim', type='int', default='300')
    p.add_option('-v', '--verbose', action='store_true',
                 dest='verbose', default=False)
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


def load_article(filename):
    """Load and parse source article

    :param filename:
    :return: (raw_sentences, tokenized_sentences), (raw_question, tokenized_question), correct_answer
    """
    with open(filename) as f:
        f.readline()  # Discard article source URL
        f.readline()  # Read out separation line

        text = f.readline()
        sentences_raw = sent_tokenize(text)
        sentences_tok = [WhitespaceTokenizer().tokenize(sent) for sent in sentences_raw]
        f.readline()  # Read out separation line

        question_raw = f.readline()
        question_tok = WhitespaceTokenizer().tokenize(question_raw)
        f.readline()  # Read out separation line

        ans = f.readline()
        return (sentences_raw, sentences_tok), (question_raw, question_tok), ans


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
        articles = open(options.article_list.strip())
    elif options.article.strip():
        articles = [options.article.strip()]
    else:
        quit()

    article_dir = options.article_dir.strip()
    if article_dir and not article_dir.endswith('/'):
        article_dir += '/'

    client = RPCClient(options.server_ip, options.server_port)

    for article in nonblank_lines(articles):
        sentences, question, answer = load_article(article_dir + article)

        q_avg = vec_avg(question[1])
        s_avg = [vec_avg(s) for s in sentences[1]]
        cos = [dist.cosine(q_avg, avg) for avg in s_avg]

        fmt = '{:<4}{:<1.6f}  {}'
        if options.verbose:
            for i, (c, sent) in enumerate(zip(cos, sentences[0])):
                print(fmt.format(i, c, sent))

        print 'ARTICLE: ', article
        print 'QUESTION:', question[0],
        print 'ANSWER:  ', answer,

        print 'GUESS:   ',
        min_dist = min(cos)
        min_idx = cos.index(min_dist)
        print(fmt.format(min_idx, min_dist, sentences[0][min_idx]))

        print
