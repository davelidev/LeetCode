class Answer(object):
'''462. Minimum Moves to Equal Array Elements II'''
    def minMoves2(nums):
        nums.sort()
        mid_val = nums[len(nums) / 2]
        return sum(abs(x - mid_val) for x in nums)