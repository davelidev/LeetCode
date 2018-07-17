class Answer(object):
'''781. Rabbits in Forest'''
    def numRabbits(answers):
        from collections import Counter
        from math import ceil
        counts = Counter(x + 1 for x in answers)
        return int(sum(ceil(float(head_count) / quantity) * quantity
                   for quantity, head_count in counts.iteritems()))