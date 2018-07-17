class Answer(object):
'''868. Transpose Matrix'''
    def transpose(A):
        return [[A[i][j] for i in range(len(A))]
                for j in range(len(A[0]))] or [[]]