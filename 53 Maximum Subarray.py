class Answer(object):'''53. Maximum Subarray'''
    def maxSubArray(nums):
        max_sum = float('-inf')
        max_ending_here = float('-inf')
        for i in range(len(nums)):
            max_ending_here = max(max_ending_here, 0) + nums[i]
            max_sum = max(max_ending_here, max_sum)
        return max_sum