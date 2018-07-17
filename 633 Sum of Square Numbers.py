class Answer(object):
'''633. Sum of Square Numbers'''
    def judgeSquareSum(c):
        a, b = 0, int(c ** (0.5))
        while a <= b:
            eq = a ** 2 + b ** 2
            if eq == c: return True
            elif eq < c: a += 1
            elif eq > c: b -= 1
        return False

    def judgeSquareSum(c):
        a, b = 0, int(c ** (0.5))
        dp = [0]
        inc = 1
        for i in range(b + 1):
            dp.append(inc + dp[-1])
            inc += 2
        while a <= b:
            eq = dp[a] + dp[b]
            if eq == c: return True
            elif eq < c: a += 1
            elif eq > c: b -= 1
        return False
            
            58. Length of Last Word
    def lengthOfLastWord(s):
        s = s.strip(' ')
        return len(s) - s.rfind(' ') - 1