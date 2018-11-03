class Answer(object):'''31. Next Permutation'''
    def nextPermutation(self, nums):
        if len(nums) <= 1: return
        l, r = 0, len(nums) - 1
        # find first decreasing pair from the right,
        # then swap it with the smallest element that's strictly greater.
        for i in xrange(len(nums) - 2, -1, -1):
            if nums[i] < nums[i+1]:
                right_greater = min(((nums[j], j) for j in range(i + 1, len(nums)) if nums[j] > nums[i]),                                     key=lambda x: (x[0], -x[1]))[1]
                nums[i], nums[right_greater] = nums[right_greater], nums[i]
                l = i + 1
                break
        # reverse the elems to the right
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l, r = l + 1, r - 1