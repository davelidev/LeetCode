class Answer(object):
'''680. Valid Palindrome II'''
    def validPalindrome(self, s):
        for i in range(len(s) / 2):
            j = len(s) - i - 1
            if s[i] != s[j]:
                s1 = s[:i] + s[i + 1:]
                s2 = s[:j] + s[j + 1:]
                return s1 == s1[::-1] or s2 == s2[::-1]
        return True