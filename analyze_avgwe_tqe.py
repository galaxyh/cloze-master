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
        correct_candidate_top1 = 0
        correct_candidate_top2 = 0
        num_candidate_entity_top1 = 0
        num_candidate_entity_tqe_top1 = 0
        num_all_entity = 0
        avgwe_ev = 0
        random_ev = 0
        avgwe_tqe_ev = 0

        records = csv.reader(f, delimiter=',')
        for record in records:
            num_question += 1
            correct_candidate_top1 += int(record[0])
            correct_candidate_top2 += max(int(record[0]), int(record[1]))
            num_candidate_entity_top1 += int(record[2])
            num_candidate_entity_tqe_top1 += int(record[6])
            num_all_entity += int(record[4])
            avgwe_ev += (float(record[0]) / int(record[2])) if int(record[2]) > 0 else 0
            random_ev += (1.0 / int(record[8])) if int(record[8]) > 0 else 0

            if int(record[6]) > 0:
                avgwe_tqe_ev += float(record[4]) / int(record[6])
            else:
                avgwe_tqe_ev += (float(record[5]) / int(record[7])) if int(record[7]) > 0 else 0

        print options.filename.strip()
        print '--'
        print 'Total questions answered =', num_question
        print 'Average number of entities in an article = {:>1.3f}'.format(float(num_all_entity) / num_question)
        print 'Correct candidate sentences (top 1) =', correct_candidate_top1
        print 'Correct candidate sentence rate (top 1) = {:>1.3f}'.format(
                float(correct_candidate_top1) / num_question)
        print 'Average number of entities in a candidate sentence (top 1) = {:>1.3f}'.format(
                float(num_candidate_entity_top1) / num_question)
        print 'Average number of entities in a candidate sentence (top 1, tqe) = {:>1.3f}'.format(
                float(num_candidate_entity_tqe_top1) / num_question)
        print
        print 'Correct candidate sentences (top 2) =', correct_candidate_top2
        print 'Correct candidate sentence rate (top 2) = {:>1.3f}'.format(
                float(correct_candidate_top2) / num_question)
        print
        print 'Expected value of correct answering (top 1):'
        print 'Average WE = {:>4.3f}'.format(avgwe_ev)
        print 'Random select = {:>4.3f}'.format(random_ev)
        print
        print 'Expected correct answering rate (top 1):'
        print 'Average WE = {:>1.3f}'.format(avgwe_ev / num_question)
        print 'Random select = {:>1.3f}'.format(random_ev / num_question)
        print
        print 'Expected correct answering rate (trim question entities):'
        print 'Average WE TQE= {:>1.3f}'.format(avgwe_tqe_ev / num_question)
