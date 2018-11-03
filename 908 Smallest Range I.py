class Answer(object):'''908. Smallest Range I'''
    def smallestRangeI(A, K):
        return max(max(A) - min(A) - 2 * K, 0)