class Answer(object):
'''168. Excel Sheet Column Title'''
    def convertToTitle(n):
        res = []
        mapping = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        while n != 0:
            res.append(mapping[(n - 1) % 26])
            n = (n - 1) / 26
        return ''.join(list(reversed(res)))