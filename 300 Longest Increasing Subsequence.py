class Answer(object):
'''300. Longest Increasing Subsequence'''
    # dp[i] :=  the index of the last number in sequence of size i
    def lengthOfLIS(self, nums):
        end_idx = [None] * len(nums)
        length = 0
        for i, num in enumerate(nums):
            j = 0
            while j < length and nums[end_idx[j]] < num:
                j += 1
            end_idx[j] = i
            length = max(j + 1, length)
        return length