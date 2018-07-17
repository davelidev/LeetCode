class Answer(object):
'''303. Range Sum Query - Immutable'''
    class NumArray(object):
        def __init__(self, nums):
            self.l_sum = l_sum = nums
            for i in range(1, len(l_sum)): l_sum[i] = l_sum[i - 1] + l_sum[i]
        def sumRange(self, i, j):
            return self.l_sum[j] - (self.l_sum[i - 1] if i > 0 else 0)