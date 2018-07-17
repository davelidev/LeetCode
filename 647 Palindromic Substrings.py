class Answer(object):
'''647. Palindromic Substrings'''
    def countSubstrings(s):
        return sum(s[i:j] == s[i:j][::-1] for j in range(len(s) + 1) for i in range(j))