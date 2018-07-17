class Answer(object):
'''419. Battleships in a Board'''
    def countBattleships(board):
        m, n = len(board), len(board[0])
        for i in range(m):
            for j in range(n):
                if i + 1 < m and 'X' == board[i + 1][j] or                     j + 1 < n and 'X' == board[i][j + 1]:
                    board[i][j] = '.'
        return sum(1 for row in board for el in row if el == 'X')