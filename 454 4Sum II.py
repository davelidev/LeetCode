class Answer(object):
'''454. 4Sum II'''
    def fourSumCount(A, B, C, D):
        from collections import Counter, defaultdict
        sum_count = defaultdict(int, Counter([a + b for a in A for b in B]))
        return sum(sum_count[-d-c] for c in C for d in D)