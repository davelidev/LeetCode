class Answer(object):
'''448. Find All Numbers Disappeared in an Array'''
    def findDisappearedNumbers(nums):
        for num in nums: nums[abs(num) - 1] = -1 * abs(nums[abs(num) - 1])
        return [i + 1 for i, num in enumerate(nums) if num > 0]