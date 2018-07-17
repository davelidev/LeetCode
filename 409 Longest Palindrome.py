class Answer(object):
'''409. Longest Palindrome'''
    def longestPalindrome(s):
        from collections import Counter
        odds_len = sum(count & 1 for count in Counter(s).values())
        return len(s) - odds_len + bool(odds_len)