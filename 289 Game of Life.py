class Answer(object):
'''289. Game of Life'''
    def gameOfLife(board):
        if not board: return
        x_y_diff = [-1, 0, 1]
        m, n = len(board), len(board[0])
        for i in range(m):
            for j in range(n):
                neighbors = [(i + x, j + y) for x in x_y_diff for y in x_y_diff if (x or y)]
                
                count_life = 0
                for x, y in neighbors:
                    if (0 <= x < m) and (0 <= y < n) and (board[x][y] in [1, 2]):
                        count_life += 1

                if board[i][j] and (count_life < 2 or count_life > 3): board[i][j] = 2
                elif not board[i][j] and count_life == 3: board[i][j] = 3

        for i in range(m):
            for j in range(n):
                board[i][j] = 1 & board[i][j]