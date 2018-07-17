class Answer(object):
'''529. Minesweeper'''
    def updateBoard(board, click):
        def get_adjs(i, j):
            dif = [-1, 0, 1]
            xy_dir = [(x, y) for x in dif for y in dif if x or y]
            adjs = [(i + x, j + y) for x, y in xy_dir]
            adjs = [(x, y) for x, y in adjs if 0 <= x < len(board) and 0 <= y < len(board[0])]
            return adjs
        def click_board(i, j):
            if board[i][j] == 'M':
                board[i][j] = 'X'
            elif board[i][j] == 'E':
                adjs = get_adjs(i, j)
                board[i][j] = str(sum(1 for x, y in adjs if board[x][y] == 'M') or 'B')
                if board[i][j] == 'B':
                    for x, y in adjs: click_board(x, y)
        click_board(*click)
        return board