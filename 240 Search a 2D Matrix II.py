class Answer(object):
'''240. Search a 2D Matrix II'''
    def searchMatrix(self, matrix, target):
        if not matrix or not any(matrix):
            return False
        col_i = len(matrix[0]) - 1
        for row in matrix:
            while row[col_i] > target and col_i > 0:
                col_i -= 1
            if row[col_i] == target:
                return True
        return False