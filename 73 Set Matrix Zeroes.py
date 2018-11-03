class Answer(object):'''73. Set Matrix Zeroes'''
    def setZeroes(matrix):
        m, n = len(matrix), len(matrix[0])
        row_1_zero = not all(matrix[0])
        col_1_zero = not all(row[0] for row in matrix)
        for i, row in enumerate(matrix):
            for j, el in enumerate(row):
                if not el: matrix[i][0] = matrix[0][j] = 0
        for i in range(1, m):
            for j in range(1, n):
                if not matrix[i][0] or not matrix[0][j]: matrix[i][j] = 0
        if row_1_zero: matrix[0] = [0] * n
        if col_1_zero:
            for i in range(m): matrix[i][0] = 0

    def setZeroes(matrix):
        m, n = len(matrix), len(matrix[0])
        # set None indicate row/col contains 0, 0 indicates the elem is 0
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    for x in range(m):
                        matrix[x][j] = None if matrix[x][j] else matrix[x][j]
                    for y in range(n):
                        matrix[i][y] = None if matrix[i][y] else matrix[i][y]
        for i in range(m):
            for j in range(n):
                matrix[i][j] = matrix[i][j] or 0