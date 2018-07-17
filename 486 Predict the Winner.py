class Answer(object):
'''486. Predict the Winner'''
    def PredictTheWinner(nums):
        dp = [[None] * (len(nums) + 1) for _ in range(len(nums) + 1)]
        def _PredictTheWinner(i, j):
            if dp[i][j] is not None:
                return dp[i][j]
            if i == j:
                dp[i][j] = nums[i], 0
            else:
                o_r, s_r = _PredictTheWinner(i, j - 1)
                o_l, s_l = _PredictTheWinner(i + 1, j)
                s_r += nums[j]
                s_l += nums[i]
                if s_r - o_r > s_l - o_l:
                    dp[i][j] =  s_r, o_r
                else:
                    dp[i][j] = s_l, o_l
            return dp[i][j]
        p1, p2 = _PredictTheWinner(0, len(nums) - 1)
        return p1 >= p2