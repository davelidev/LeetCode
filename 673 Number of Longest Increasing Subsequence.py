class Answer(object):
'''673. Number of Longest Increasing Subsequence'''
    def lengthOfLIS(nums):
        end_idx = [None] * len(nums)
        length = 0
        for i, num in enumerate(nums):
            j = 0
            while j < length and nums[end_idx[j]] < num:
                j += 1
            end_idx[j] = i
            length = max(j + 1, length)
        return length