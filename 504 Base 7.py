class Answer(object):
'''504. Base 7'''
    def convertToBase7(num):
        base = 7
        res = ""
        neg_prefix = '-' if num < 0 else ''
        num = abs(num)
        while num != 0:
            num, mod = divmod(num, base)
            res = str(mod) + res
        return (neg_prefix + res) or '0'