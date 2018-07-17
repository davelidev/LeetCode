class Answer(object):
'''413. Arithmetic Slices'''
    def numberOfArithmeticSlices(A):
        diffs = []
        prev = None
        for i in range(1, len(A)):
            diff = A[i] - A[i - 1]
            if not diffs or prev != diff:
                diffs.append(1)
            else: diffs[-1] += 1
            prev = diff
        return sum((n * (n - 1) / 2) for n in diffs)