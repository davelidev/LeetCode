class Answer(object):
'''525. Contiguous Array'''
    def findMaxLength(nums):
        for i in range(len(nums)): nums[i] = -1 if nums[i] == 0 else 1
        sum_to_idx, sum_so_far, max_len = {0: -1}, 0, 0
        for i in range(len(nums)):
            sum_so_far += nums[i]
            if sum_so_far in sum_to_idx:
                max_len = max(i - sum_to_idx[sum_so_far], max_len)
            sum_to_idx.setdefault(sum_so_far, i)
        return max_len