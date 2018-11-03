class Answer(object):'''74. Search a 2D Matrix'''
    def searchMatrix(matrix, target):
        def _searchMatrix(start, end):
            if end <= 0:
                return False
            elif end == start + 1:
                return matrix[start/len(matrix[0])][start%len(matrix[0])] == target
            else:
                mid = (start + end) / 2
                mid_val = matrix[mid/len(matrix[0])][mid%len(matrix[0])]
                if mid_val == target:
                    return True
                elif mid_val < target:
                    return _searchMatrix(mid, end)
                else:
                    return _searchMatrix(start, mid)
        if not matrix:
            return False
        return _searchMatrix(0, len(matrix[0]) * len(matrix))