class Answer(object):
'''343. Integer Break'''
    def integerBreak(n):
        dp = [None] * (n + 1)
        dp[1] = 1
        if n <= 3: return n - 1
        def get_max(n):
            from itertools import chain
            if dp[n] is not None: return dp[n]
            dp[n] = max(chain((get_max(i) * get_max(n-i) for i in range(1, n)), (n, )))
            return dp[n]
        return get_max(n)