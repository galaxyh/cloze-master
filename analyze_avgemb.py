import csv
import codecs
from optparse import OptionParser


def parse_args():
    p = OptionParser()
    p.add_option('-f', '--filename', action='store',
                 dest='filename', type='string', default='')
    return p.parse_args()

if __name__ == '__main__':
    options, remainder = parse_args()

    if options.filename.strip():
        quit()

    with codecs.open(options.filename.strip(), 'r', encoding='utf8') as f:
        avgemb_ev = 0
        random_ev = 0
        records = csv.reader(f, delimiter=',')
        for record in records:
            avgemb_ev += (1.0 / record[1]) if record[1] > 0 else 0
            random_ev += (1.0 / record[2]) if record[2] > 0 else 0

        print 'Expected value of correct answer'
        print 'Average embedding = {.3f}'.format(avgemb_ev)
        print 'Random select     = {.3f}'.format(random_ev)
