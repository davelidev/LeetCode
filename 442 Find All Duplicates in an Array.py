class Answer(object):
'''442. Find All Duplicates in an Array'''
    def findDuplicates(nums):
        res = []
        for i in range(len(nums)):
            idx = abs(nums[i]) - 1
            if nums[idx] < 0: res.append(idx + 1)
            else: nums[idx] = -nums[idx]
        return res