class Answer(object):
'''171. Excel Sheet Column Number'''
    def titleToNumber(self, s):
        return reduce(lambda x, y: (ord(y) - ord('a') + 1) + (x * 26), s.lower(), 0)