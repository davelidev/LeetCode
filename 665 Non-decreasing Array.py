class Answer(object):
'''665. Non-decreasing Array'''
    def checkPossibility(nums):
        modified = False
        for i in range(1, len(nums)):
            if nums[i - 1] > nums[i]:
                if modified: return False
                if i - 2 < 0 or nums[i - 2] <= nums[i]: nums[i - 1] = nums[i]
                else: nums[i] = nums[i - 1]
                modified = True
        return True