class Answer(object):
'''494. Target Sum'''
    def findTargetSumWays(nums, S):
        dp = {0: 1}
        for num in nums:
            new_dp = {}
            for key, val in dp.iteritems():
                for new_key in [key + num, key - num]:
                    new_dp.setdefault(new_key, 0)
                    new_dp[new_key] += val
            dp = new_dp
        return dp.get(S, 0)