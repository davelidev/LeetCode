class Answer(object):
'''540. Single Element in a Sorted Array'''
    def singleNonDuplicate(nums):
        res = 0
        for num in nums: res ^= num
        return res