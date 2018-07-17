class Answer(object):
'''190. Reverse Bits'''
    def reverseBits(n):
        res = 0
        for i in range(32):
            n, mod = divmod(n, 2)
            res = (res << 1) | mod
        return res