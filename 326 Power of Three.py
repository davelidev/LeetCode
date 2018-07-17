class Answer(object):
'''326. Power of Three'''
    def isPowerOfThree(n):
        # import sys
        # power = 1
        # while (3 ** (power + 1)) <= sys.maxint:
        #     power += 1
        # print 3 ** power
        return (n > 0) and (4052555153018976267 % n == 0)