class Answer(object):'''884. Uncommon Words from Two Sentences'''
    def uncommonFromSentences(A, B):
        from collections import Counter
        A = A.split(' ')
        B = B.split(' ')
        A = Counter(A)
        B = Counter(B)
        return [a for a, count in A.iteritems() if count == 1 and a not in B] +             [b for b, count in B.iteritems() if count == 1 and b not in A]