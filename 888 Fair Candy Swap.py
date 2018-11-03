class Answer(object):'''888. Fair Candy Swap'''
    def fairCandySwap(A, B):
        n, m = sum(A), sum(B)
        gap = (m - n) / 2
        B_set = set(B)
        for i in A:
            if (i + gap) in B_set:
                return (i, (i + gap))