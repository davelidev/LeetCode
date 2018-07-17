class Answer(object):
'''139. Word Break'''
    def wordBreak(self, s, wordDict):
        dp = [False] * (len(s) + 1)
        dp[0] = True
        wordDict = set(wordDict)
        for j in range(1, len(s) + 1):
            for i in range(j - 1, -1, -1):
                if dp[i] and s[i: j] in wordDict:
                    dp[j] = True
                    continue
        return dp[-1]