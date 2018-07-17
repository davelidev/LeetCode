class Answer(object):
'''521. Longest Uncommon Subsequence I'''
    def findLUSlength(a, b):
        return -1 if a == b else max(len(a), len(b))