class Answer(object):
'''268. Missing Number'''
    def missingNumber(nums):
        n = len(nums) + 1
        return (n * (n - 1)) / 2 - sum(nums)

    def missingNumber(nums):
        for i in range(len(nums)): nums[i] += 1
        nums.append(1)
        nums.append(1)
        for i in range(len(nums) - 2): nums[abs(nums[i])] = -nums[abs(nums[i])]
        for i, num in enumerate(nums):
            if i != 0 and num > 0:
                return i - 1