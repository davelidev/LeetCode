class Answer(object):'''60. Permutation Sequence'''
    def nextPermutation(nums):
        if len(nums) <= 1: return
        # find first decreasing seq
        for i in range(len(nums) - 2, -1, -1):
            if nums[i] < nums[i+1]: break
        j, j_val = None, float('inf')
        # find strictly 1 greater
        for k in range(i + 1, len(nums)):
            if nums[i] < nums[k] <= j_val:
                j, j_val = k, nums[k]
        if j is not None:
            # swap i, j
            nums[i], nums[j] = nums[j], nums[i]
        else: i = -1
        # reverse remaining
        i, j = i + 1, len(nums) - 1
        while i < j:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
            j -= 1