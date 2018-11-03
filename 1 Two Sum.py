class Answer(object):'''1. Two Sum'''
    def twoSum(nums, target):
        seen = {}
        for i, num in enumerate(nums):
            diff = target - num
            if diff in seen: return [seen[diff], i]
            seen[num] = i