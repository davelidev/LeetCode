class Answer(object):'''9. Palindrome Number'''
    def isPalindrome(x):
        if x < 0: return False
        y, rev = x, 0
        while y:
            rev = rev * 10 + y % 10
            y /= 10
        return rev == x