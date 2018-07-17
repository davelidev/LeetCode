class Answer(object):
'''77. Combinations'''
    def combine(n, k):
        cur, res = [], []
        def _combinations(i):
            if len(cur) == k: return res.append(cur[:])
            for j in range(i, n + 1):
                cur.append(j)
                _combinations(j + 1)
                cur.pop()
        _combinations(1)
        return res