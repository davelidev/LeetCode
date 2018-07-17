class Answer(object):
'''643. Maximum Average Subarray I'''
    def findMaxAverage(nums, k):
        for i in range(1, len(nums)): nums[i] += nums[i - 1]
        max_avg = float('-inf')
        for i in range(k - 1, len(nums)):
            left, right = (0 if i - k < 0 else nums[i - k]), float(nums[i])
            max_avg = max(max_avg, (right - left) / k)
        return max_avg