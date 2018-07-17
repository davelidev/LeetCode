class Answer(object):
'''120. Triangle'''
    def minimumTotal(triangle):
        for i in range(len(triangle) - 2, -1, -1):
            row = triangle[i]
            next_row = triangle[i + 1]
            for j in range(len(row)):
                row[j] = min(next_row[j], next_row[j + 1]) + row[j]
        return triangle[0][0] if triangle else 0