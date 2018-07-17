class Answer(object):
'''191. Number of 1 Bits'''
    def hammingWeight(n):
        count = 0
        while n:
            count += 1 & n
            n = n >> 1
        return count