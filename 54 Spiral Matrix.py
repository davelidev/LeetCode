class Answer(object):'''54. Spiral Matrix'''
    def spiralOrder(matrix):
        if not matrix:
            return []
        i = 0
        res = []
        a, b, c, d = 0, len(matrix[0]) - 1, len(matrix) - 1, 0
        while a <= c and b >= d:
            if i % 4 == 0:
                for j in range(d, b + 1):
                    res.append(matrix[a][j])
                a += 1
            elif i % 4 == 1:
                for j in range(a, c + 1):
                    res.append(matrix[j][b])
                b -= 1
            elif i % 4 == 2:
                for j in range(b, d - 1, -1):
                    res.append(matrix[c][j])
                c -= 1
            elif i % 4 == 3:
                for j in range(c, a - 1, -1):
                    res.append(matrix[j][d])
                d += 1
            i += 1
        return res