class Answer(object):
'''304. Range Sum Query 2D - Immutable'''
    class NumMatrix(object):

        def __init__(self, matrix):
            self.matrix = matrix
            for i in range(len(matrix)):
                for j in range(len(matrix[0]) if matrix else 0):
                    matrix[i][j] = self._sub_points([[i - 1, j], [i, j - 1], [i, j]], [[i - 1, j - 1]])
        
        def _sub_points(self, coords, sub_coords):
            def sum_points(coords):
                return sum(self.matrix[x][y] for x, y in coords if x >=0 and y >= 0)
            return sum_points(coords) - sum_points(sub_coords)
                    
        def sumRegion(self, row1, col1, row2, col2):
            return self._sub_points([[row2, col2], [row1 - 1, col1 - 1]],[[row1 - 1, col2], [row2, col1 - 1]])