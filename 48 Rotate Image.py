class Answer(object):'''48. Rotate Image'''
    def rotate(matrix):
        for i in range(len(matrix) / 2):
            for j in range(i, len(matrix) - i - 1):
                n = len(matrix) - 1
                vals = matrix[i][j], matrix[j][n - i], matrix[n - i][n - j], matrix[n - j][i]
                matrix[i][j], matrix[j][n - i], matrix[n - i][n - j], matrix[n - j][i] = vals[3], vals[0], vals[1], vals[2]

    def rotate(self, matrix):
        '''Reverse => transpose'''
        n = len(matrix) - 1
        for i in range(len(matrix) / 2): matrix[i], matrix[n-i] = matrix[n-i], matrix[i]
        for i in range(n): for j in range(i + 1, n + 1): matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]