class Answer(object):
'''130. Surrounded Regions'''
    def solve(board):
        if not board: return
        m, n = len(board), len(board[0])
        bfs = [pair for x in range(m) for pair in [(x, 0),(x, n - 1)]] +               [pair for y in range(n) for pair in [(0, y),(m - 1, y)]]
        
        while bfs:
            x, y = bfs.pop()
            if 0 <= x < m and 0 <= y < n and board[x][y] == 'O':
                board[x][y] = ''
                bfs.extend([x + a, y + b] for a, b in zip([0,1,0,-1], [1,0,-1,0]))

        for i in range(m):
            for j in range(n):
                board[i][j] = 'X' if board[i][j] else 'O'