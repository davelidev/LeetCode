class Answer(object):
'''198. House Robber'''
    def rob(nums):
        for i in range(1, len(nums)):
            nums[i] = max([nums[i - 1], nums[i]] if i - 2 < 0 else [nums[i - 1], nums[i - 2] + nums[i]])
        return nums[-1] if nums else 0