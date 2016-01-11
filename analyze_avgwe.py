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

    if not options.filename.strip():
        quit()

    with codecs.open(options.filename.strip(), 'r', encoding='utf8') as f:
        num_question = 0
        correct_candidate = 0
        num_candidate_entity = 0
        num_all_entity = 0
        avgwe_ev = 0
        random_ev = 0
        records = csv.reader(f, delimiter=',')
        for record in records:
            num_question += 1
            correct_candidate += int(record[0])
            num_candidate_entity += int(record[1])
            num_all_entity += int(record[2])
            avgwe_ev += (1.0 / int(record[1])) if int(record[1]) > 0 else 0
            random_ev += (1.0 / int(record[2])) if int(record[2]) > 0 else 0

        print options.filename.strip()
        print '--'
        print 'Total questions answered =', num_question
        print 'Correct candidate sentences =', correct_candidate
        print 'Correct candidate sentences rate = {:>1.3f}'.format(float(correct_candidate) / num_question)
        print 'Average number of entities in a question = {:>1.3f}'.format(float(num_all_entity) / num_question)
        print 'Average number of entities in a candidate sentence = {:>1.3f}'.format(
            float(num_candidate_entity) / num_question)
        print
        print 'Expected value of correct answering:'
        print 'Average WE = {:>4.3f}'.format(avgwe_ev)
        print 'Random select = {:>4.3f}'.format(random_ev)
        print
        print 'Expected correct answering rate:'
        print 'Average WE = {:>1.3f}'.format(avgwe_ev / num_question)
        print 'Random select = {:>1.3f}'.format(random_ev / num_question)
