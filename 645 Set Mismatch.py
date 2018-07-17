class Answer(object):
'''645. Set Mismatch'''
    def findErrorNums(nums):
        for num in nums:
            if nums[abs(num) - 1] < 0: 
                dup = abs(num)
                break
            nums[abs(num) - 1] = -nums[abs(num) - 1]
        for i in range(len(nums)): nums[i] = abs(nums[i])
        n = len(nums)
        return dup, ((n * (n + 1) / 2)) - sum(nums) + dup