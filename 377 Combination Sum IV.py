class Answer(object):
'''377. Combination Sum IV'''
    def combinationSum4(nums, target):
        dp = {0: 1}
        def _combinationSum4(target):
            if target in dp: return dp[target]
            res = 0
            for num in nums:
                if target - num >= 0:
                    res += _combinationSum4(target - num)
            dp[target] = res
            return res
        return _combinationSum4(target)