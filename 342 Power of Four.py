class Answer(object):
'''342. Power of Four'''
    def isPowerOfFour(num):
        # mask = 0
        # while (mask << 2 | 1) < ((1 << 31) - 1):
        #     mask = mask << 2 | 1
        # print mask
        return (num & (num - 1)) == 0 and bool(1431655765 & num)