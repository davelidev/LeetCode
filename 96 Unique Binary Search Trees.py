class Answer(object):
'''96. Unique Binary Search Trees'''
    def numTrees(self, n):
        dp = [1, 1]
        if n < len(dp):
            return dp[n]
        for i in range(2, n + 1):
            next_val = 0
            for j in range(1, i + 1):
                next_val += (dp[j - 1]) * (dp[i - j])
            dp.append(next_val)
        return dp[-1]