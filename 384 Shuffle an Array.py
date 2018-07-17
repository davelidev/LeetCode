class Answer(object):
'''384. Shuffle an Array'''
    from random import randint
    class Solution(object):
        def __init__(self, nums):
            self.nums = nums
        def reset(self):
            return self.nums
        def shuffle(self):
            self.rand_nums = self.nums[:]
            for i in range(len(self.rand_nums)):
                swap_idx = randint(0, len(self.rand_nums) - 1)
                self.rand_nums[i], self.rand_nums[swap_idx] = self.rand_nums[swap_idx], self.rand_nums[i]
            return self.rand_nums