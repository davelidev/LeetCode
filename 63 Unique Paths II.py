class Answer(object):'''63. Unique Paths II'''
    def uniquePathsWithObstacles(obstacleGrid):
        if not obstacleGrid or not obstacleGrid[0]: return 0
        dp = [[1 - item for item in row] for row in obstacleGrid]
        for i in range(1, len(dp)): dp[i][0] = min(dp[i - 1][0], dp[i][0])
        for j in range(1, len(dp[0])): dp[0][j] = min(dp[0][j - 1], dp[0][j])
        for i in range(1, len(dp)):
            for j in range(1, len(dp[0])):
                if dp[i][j]: dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]