class Answer(object):
'''221. Maximal Square'''
    def maximalSquare(matrix):
        from itertools import chain
        m, n = len(matrix), len(matrix[0]) if matrix else 0
        matrix = [map(int, row) for row in matrix]
        # check if the first row or colum contains a 1
        max_w = any(chain((matrix[0] if matrix else []), (row[0] for row in matrix)))
        for i in range(1, m):
            for j in range(1, n):
                min_wh = min(matrix[i - 1][j], matrix[i][j - 1])
                is_inc = matrix[i - min_wh][j - min_wh] and matrix[i][j]
                matrix[i][j] = (min_wh if matrix[i][j] else 0) + is_inc
                max_w = max(max_w, matrix[i][j])
        return max_w ** 2