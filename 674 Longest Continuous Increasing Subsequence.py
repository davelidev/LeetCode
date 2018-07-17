class Answer(object):
'''674. Longest Continuous Increasing Subsequence'''
    def findLengthOfLCIS(nums):
        nums.append(float('-inf'))
        i = max_len = 0
        for j in range(1, len(nums)):
            if nums[j-1] >= nums[j]:
                max_len = max(max_len, j - i)
                i = j
        return max_len