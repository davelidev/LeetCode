class Answer(object):'''832. Flipping an Image'''
    def flipAndInvertImage(A):
        return [map(lambda x: 1 - x, reversed(lst)) for lst in A]