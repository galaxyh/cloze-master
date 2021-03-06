import multiprocessing
import time
import logging


def run_word2vec_server(is_ready):
    import gensim
    from gevent.server import StreamServer
    from mprpc import RPCServer

    class Word2VecServer(RPCServer):
        def __init__(self, *args, **kwargs):
            super(Word2VecServer, self).__init__(*args, **kwargs)
            is_ready.value = False
            self.model = gensim.models.Word2Vec.load_word2vec_format(kwargs['model_file'], binary=True)
            is_ready.value = True

        def most_similar(self, positive, negative, topn, restrict_vocab):
            return self.model.most_similar(positive=positive, negative=negative, topn=topn, restrict_vocab=restrict_vocab)

        def doesnt_match(self, words):
            return self.model.doesnt_match(words)

        def similarity(self, w1, w2):
            return self.model.similarity(w1, w2)

        def vector(self, word):
            return self.model[word].tolist()

        def exist(self, word):
            return word in self.model

    options, remainder = parse_args()
    logger.info('Server starting...')
    server = StreamServer((options.server_ip, options.server_port), Word2VecServer(model_file=options.model_file))
    logger.info('Server started at %s:%d', options.server_ip, options.server_port)
    logger.info('Word2Vec model file name: %s', options.model_file)
    server.serve_forever()


def parse_args():
    from optparse import OptionParser
    p = OptionParser()
    p.add_option('-s', '--server_ip', action='store',
                 dest='server_ip', type='string', default='127.0.0.1')
    p.add_option('-p', '--server_port', action='store',
                 dest='server_port', type='int', default='1979')
    p.add_option('-m', '--model_file', action='store',
                 dest='model_file', type='string')
    return p.parse_args()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger('word2vec_server')

    ready = multiprocessing.Value('b', False)
    daemon = multiprocessing.Process(target=run_word2vec_server, args=(ready,))
    daemon.daemon = True
    daemon.start()

    while not ready.value:
        time.sleep(1)

    print
    print 'Enter \'quit\' to stop server.'
    print

    while True:
        s = raw_input('Word2Vec > ')
        if s and s.lower().strip() == 'quit':
            break

    print
    daemon.terminate()
    logger.info('Stopping server...')
    print
    daemon.join()
    logger.info('Server stopped')
    print
