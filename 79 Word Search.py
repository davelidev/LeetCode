class Answer(object):
'''79. Word Search'''
    def exist(board, word):
        m, n = len(board), len(board[0])
        visited = [[False] * n for _ in range(m)]
        def exist(i, j, char_i):
            if word[char_i] != board[i][j]: return False
            elif char_i == len(word) - 1 and not visited[i][j]: return True
            adjs = [(i + a, j + b)
                    for a, b in zip([1,0,-1,0], [0,1,0,-1])
                    if (0 <= i + a < m) and (0 <= j + b < n)]
            visited[i][j] = True
            if any(not visited[x][y] and exist(x, y, char_i + 1)
                   for x, y in adjs):
                    return True
            visited[i][j] = False
            return False

        return any(exist(i, j, 0)
                   for i in range(m)
                   for j in range(n))