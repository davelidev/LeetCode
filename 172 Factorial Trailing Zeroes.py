class Answer(object):
'''172. Factorial Trailing Zeroes'''
    def trailingZeroes(n):
        # factors of 5 determins number of zeros
        # ..5..10..15..20..25..30
        # ..5...5...5..5..5*5...5
        # i = 1: n/5 -> 6
        # ...1...2...3..4....5...6
        # i = 2: n/5 -> 1
        # ...1...2...3..4....1...6
        # res = 6 + 1 = 7
        res = 0
        while n:
            res += n / 5
            n /= 5
        return res