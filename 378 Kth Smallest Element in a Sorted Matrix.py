class Answer(object):
'''378. Kth Smallest Element in a Sorted Matrix'''
    def kthSmallest(matrix, k):
        return sorted(i for row in matrix for i in row)[k-1]