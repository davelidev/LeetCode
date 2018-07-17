class Answer(object):
'''131. Palindrome Partitioning'''
    def partition(s):
        n = len(s) + 1
        dp = [[[]]]
        for j in range(1, n):
            dp.append(
                [prefix + [s[i:j]]
                 for i in range(j)
                 if s[i:j] == s[i:j][::-1]
                 for prefix in dp[i]])
        return dp[-1]