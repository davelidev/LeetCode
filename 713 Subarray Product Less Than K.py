class Answer(object):
'''713. Subarray Product Less Than K'''
    def numSubarrayProductLessThanK(nums, k):
        if k == 0 : return 0
        prod = 1
        start = count = 0
        for end, elem in enumerate(nums):
            prod *= elem
            while prod >= k and start <= end:
                prod /= nums[start]
                start += 1
            count += end - start + 1
        return count