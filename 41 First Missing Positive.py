class Answer(object):'''41. First Missing Positive'''
    def firstMissingPositive(nums):
        # index visited is marked as negative
        # pass1: make negative nums positive, and make it len(nums) + 1 so it does not influence our algorithm
        # pass2: mark all the indicies visited as negative
        # pass3: find the first positive
        for i, num in enumerate(nums):
            if num <= 0: nums[i] = len(nums) + 1
        nums.append(len(nums) + 1)
        for i, num in enumerate(nums):
            if 0 < abs(num) < len(nums): nums[abs(num)] = -abs(nums[abs(num)])
        return next((i for i, num in enumerate(nums) if i and num > 0), len(nums))