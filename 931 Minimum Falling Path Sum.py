class Answer(object):'''931. Minimum Falling Path Sum'''
    def minFallingPathSum(A):
        prev = A[0][:]
        row = []
        n = len(A)
        for i in range(1, n):
            row = []
            for j in range(n):
                cur = min((prev[j - 1] if (j - 1) >= 0 else float('inf')),
                          (prev[j + 1]  if (j + 1) < n else float('inf')),
                          prev[j])
                row.append(cur + A[i][j])
            prev = row
        return min(row or prev)