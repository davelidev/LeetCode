class Answer(object):
'''766. Toeplitz Matrix'''
    def isToeplitzMatrix(matrix):
        m, n = len(matrix), len(matrix[0])
        for i in range(m):
            for j in range(n):
                if i and j and matrix[i - 1][j - 1] != matrix[i][j]:
                    return False
        return True