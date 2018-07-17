class Answer(object):
'''565. Array Nesting'''
    def arrayNesting(nums):
        max_dep = 0
        for i in range(len(nums)):
            cur, depth = i, 0
            while nums[cur] is not None:
                nums[cur], cur = None, nums[cur]
                depth += 1
            max_dep = max(depth, max_dep)
        return max_dep