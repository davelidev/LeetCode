class Answer(object):
'''611. Valid Triangle Number'''
    def triangleNumber(nums):
        res = 0
        nums.sort()
        res = 0
        for i in range(len(nums) -1, 1, -1):
            l, r = 0, i - 1
            while l < r:
                if nums[l] + nums[r] > nums[i]:
                    res += (r - l)
                    r -= 1
                else:
                    l += 1
        return res