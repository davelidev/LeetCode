class Answer(object):
'''87. Scramble String'''
    def isScramble(s1, s2):
        dp = {}
        from collections import Counter
        def _isScramble(s1, s2):
            if s1 == s2: return True
            elif (s1, s2) in dp: return dp[(s1, s2)]
            elif sorted(s1) != sorted(s2):
                dp[s1, s2] = False
                return False
            n, f = len(s1), _isScramble
            for i in range(1, len(s1)):
                if f(s1[i:], s2[i:]) and f(s1[:i], s2[:i]) or                    f(s1[i:], s2[:-i]) and f(s1[:i], s2[-i:]):
                    dp[(s1, s2)] = True
                    return True
            dp[(s1, s2)] = False
            return False
        return _isScramble(s1, s2)