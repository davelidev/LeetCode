class Answer(object):'''7. Reverse Integer'''
     def reverse(x):
        sign = 1 if x >= 0 else -1
        x = abs(x)
        res = 0
        while x:
            res = res * 10 + x % 10
            x /= 10
        return res * sign * (0 if res >> 31 else 1)