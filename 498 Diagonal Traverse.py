class Answer(object):
'''498. Diagonal Traverse'''
    def findDiagonalOrder(matrix):
        if not matrix or not all(matrix): return []
        res = []
        def get_diag(i, j):
            new_diag = []
            while i >= 0 and j < len(matrix[0]):
                new_diag.append(matrix[i][j])
                i -= 1
                j += 1
            res.append(new_diag)
        for i in range(len(matrix)): get_diag(i, 0)
        for j in range(1, len(matrix[0])): get_diag(len(matrix) - 1, j)
        for i in range(1, len(res), 2): res[i] = list((reversed(res[i])))
        return [item for lst in res for item in lst]