class Answer(object):
'''392. Is Subsequence'''
    def isSubsequence(s, t):
        i = 0
        for char in s:
            while i < len(t) and t[i] != char:
                i += 1
            i += 1
            if i > len(t):
                return False
        return True