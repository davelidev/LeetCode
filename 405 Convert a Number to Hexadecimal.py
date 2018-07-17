class Answer(object):
'''405. Convert a Number to Hexadecimal'''
    def toHex(num):
        if not num: return '0'
        res = ''
        mask = reduce(lambda x,y: x | (1 << y), range(4), 0)
        mapping = '0123456789abcdef'
        for _ in range(8):
            res = mapping[(num & mask) % len(mapping)] + res
            num >>= 4
        return res.lstrip('0')