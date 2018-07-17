class Answer(object):
'''459. Repeated Substring Pattern'''
    def repeatedSubstringPattern(s): return s in (s + s)[1:-1]