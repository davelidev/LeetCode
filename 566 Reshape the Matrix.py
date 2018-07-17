class Answer(object):
'''566. Reshape the Matrix'''
    def matrixReshape(nums, r, c):
        if r * c != len(nums) * len(nums[0]): return nums
        res = [num for row in nums for num in row]
        return [res[i:i+c] for i in range(0, len(res), c)]