class Answer(object):
'''152. Maximum Product Subarray'''
    def maxProduct(nums):
        large = small = max_val = nums[0]
        for i in range(1, len(nums)):
            num = nums[i]
            vals = [num, small * num, large * num]
            small, large = min(vals), max(vals)
            max_val = max(large, max_val)
        return max_val