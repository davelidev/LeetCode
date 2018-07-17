class Answer(object):
'''306. Additive Number'''
    def isAdditiveNumber(num):
        def is_seq(i, j, k):
            if k == len(num): return True
            a, b = int(num[i:j]), int(num[j:k])
            if len(str(a)) != j - i or len(str(b)) != k - j: return False
            total = str(a + b)
            return num[k:].startswith(total) and is_seq(j, k, k + len(total))
        return any(is_seq(0, i, j)
                   for j in range(2, len(num))
                   for i in range(1, j))