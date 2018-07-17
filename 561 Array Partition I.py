class Answer(object):
'''561. Array Partition I'''
    def arrayPairSum(nums):
        nums.sort()
        return sum(nums[::2])