class Answer(object):
'''118. Pascal's Triangle'''
    def generate(numRows):
        tri = []
        for i in range(1, numRows + 1):
            for j in range(i):
                if j == 0: tri.append([1])
                elif j == i - 1: tri[-1].append(1)
                else: tri[-1].append(sum(tri[-2][j - 1: j + 1]))
        return tri