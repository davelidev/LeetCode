class Answer(object):
'''213. House Robber II'''
    def rob(nums):
        def _rob(nums):
            for i in range(1, len(nums)):
                nums[i] = max([nums[i - 1], nums[i]] if i - 2 < 0 else [nums[i - 1], nums[i - 2] + nums[i]])
            return nums[-1] if nums else 0
        return max(_rob(nums[:-1]), _rob(nums[1:])) if len(nums) > 1 else (nums or [0]) [0]