class Answer(object):
'''329. Longest Increasing Path in a Matrix'''
    def longestIncreasingPath(matrix):
        xy_dir = zip([-1, 0, 1, 0], [0, -1, 0, 1])
        m, n = len(matrix), len(matrix[0]) if matrix else 0
        dp = [[None] * n for _ in range(m)]
        def get_max_inc(x, y):
            if dp[x][y] is not None: return dp[x][y]
            adj = [(i, j) for i, j in [(x + x_d, y + y_d) for x_d, y_d in xy_dir]
                   if 0 <= i < m and 0 <= j < n and matrix[x][y] > matrix[i][j]]
            dp[x][y] = max([(get_max_inc(i, j) + 1) for i, j in adj] or [1])
            return dp[x][y]
        return max([get_max_inc(x, y) for x in range(m) for y in range(n)] or [0])