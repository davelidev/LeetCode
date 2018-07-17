class Answer(object):
'''231. Power of Two'''
    def isPowerOfTwo(n):
        return (n > 0) and (n & (n - 1)) == 0