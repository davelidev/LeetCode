class Answer(object):
'''667. Beautiful Arrangement II'''
    def constructArray(n, k):
        if k == 1: return range(1, n + 1)
        res = []
        i, j = 1, k + 1
        while i <= j:
            res.extend([j, i])
            i += 1
            j -= 1
        if len(res) != k + 1: res.pop()
        res.extend(range(k + 2, n + 1))
        return res