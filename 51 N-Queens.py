class Answer(object):'''51. N-Queens'''
    def solveNQueens(n):
        vertical, diag1, diag2 = [[False] * (2 * n) for _ in range(3)]
        cur = [['.'] * n for _ in range(n)]
        res = []
        def _solveNQueens(y):
            if y >= n: return
            for x in range(0, n):
                diag1_i, diag2_i = (x - y) % (2 * n), x + y
                if not vertical[x] and not diag1[diag1_i] and not diag2[diag2_i]:
                    cur[x][y] = 'Q'
                    vertical[x] = diag1[diag1_i] = diag2[diag2_i] = True
                    if y == n - 1: res.append([''.join(row) for row in cur])
                    else: _solveNQueens(y + 1)
                    vertical[x] = diag1[diag1_i] = diag2[diag2_i] = False
                    cur[x][y] = '.'
        _solveNQueens(0)
        return res