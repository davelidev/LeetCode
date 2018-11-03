class Answer(object):'''8. String to Integer (atoi)'''
    def myAtoi(s):
        s = s.lstrip(' ')
        sign = 1
        if s.startswith('-'): s, sign = s[1:], -1
        elif s.startswith('+'): s = s[1:]
        s = s[:next((i for i, num in enumerate(s) if not num.isdigit()), len(s))]
        if not s: return 0
        int_rep = reduce(lambda x, y: x * 10 + (ord(y) - ord('0')), s, 0)
        return max(min(int_rep * sign, 2147483647), -2147483648)