class Answer(object):'''52. N-Queens II'''
    def totalNQueens(n):
        vertical, diag1, diag2 = [[False] * (2 * n) for _ in range(3)]
        self.count = 0
        def _totalNQueens(y):
            if y >= n: return
            for x in range(0, n):
                diag1_i, diag2_i = (x - y) % (2 * n), x + y
                if not vertical[x] and not diag1[diag1_i] and not diag2[diag2_i]:
                    vertical[x] = diag1[diag1_i] = diag2[diag2_i] = True
                    if y == n - 1: self.count += 1
                    else: _totalNQueens(y + 1)
                    vertical[x] = diag1[diag1_i] = diag2[diag2_i] = False
        _totalNQueens(0)
        return self.count