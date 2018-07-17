class Answer(object):
'''696. Count Binary Substrings'''
    def countBinarySubstrings(s):
        s = map(len, s.replace('01', '0 1').replace('10', '1 0').split())
        return sum([min(s[i], s[i + 1]) for i in range(len(s) - 1)])