class Answer(object):
'''125. Valid Palindrome'''
    def isPalindrome(s):
        s = [char.lower() for char in s if char.isalpha() or char.isdigit()]
        return s == s[::-1]