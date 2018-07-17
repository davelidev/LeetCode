class Answer(object):
'''581. Shortest Unsorted Continuous Subarray'''
    def findUnsortedSubarray(nums):
        sorted_nums = sorted(nums)
        i, j = 0, len(nums) - 1
        while i < j:
            if nums[i] == sorted_nums[i]: i += 1
            elif nums[j] == sorted_nums[j]: j -= 1
            else: return j - i + 1
        return 0