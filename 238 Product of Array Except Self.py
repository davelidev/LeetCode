class Answer(object):
'''238. Product of Array Except Self'''
    def productExceptSelf(nums):
        dp = [1]
        for num in reversed(nums): dp.append(num * dp[-1])
        dp = dp[::-1]
        mul_so_far = 1
        res = []
        for i in range(len(nums)):
            res.append(mul_so_far * dp[i + 1])
            mul_so_far *= nums[i]
        return res