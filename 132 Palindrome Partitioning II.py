class Answer(object):
'''132. Palindrome Partitioning II'''
    def minCut(s):
        n = len(s)
        is_pal = [[None] * (n + 1) for _ in range(n + 1)]
        for i in range(n): is_pal[i][i] = is_pal[i][i + 1] = True
        for k in range(2, n + 1):
            for i in range(n - k + 1):
                j = i + k
                is_pal[i][j] = s[i] == s[j - 1] and is_pal[i + 1][j - 1]
        dp = [float('inf')] * (n + 1)
        dp[0] = -1
        for j in range(n + 1):
            if j: dp[j] = dp[j - 1] + 1
            for i in range(j):
                if is_pal[i][j]: dp[j] = min(dp[j], dp[i] + 1)
        return dp[-1]