class Answer(object):
'''201. Bitwise AND of Numbers Range'''
    def rangeBitwiseAnd(m, n):
        res = ~0
        while ((m & res) != (n & res)):
            res = res << 1
        return res & m