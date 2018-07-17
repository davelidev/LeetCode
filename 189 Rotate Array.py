class Answer(object):
'''189. Rotate Array'''
    def rotate(nums, k):
        def reverse(i, j):
            while i < j:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
                j -= 1
        k %= len(nums)
        reverse(len(nums) - k, len(nums) - 1)
        reverse(0, len(nums) - k - 1)
        reverse(0, len(nums) - 1)