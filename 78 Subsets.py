class Answer(object):
'''78. Subsets'''
    def subsets(nums):
        res = [[]]
        for num in nums:
            res.extend([item + [num] for item in res])
        return res
    def subsets(nums): return [[nums[j] for j in range(len(nums)) if i & (1 << j)] for i in range(1 << len(nums))]