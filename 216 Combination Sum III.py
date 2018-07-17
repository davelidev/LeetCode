class Answer(object):
'''216. Combination Sum III'''
    def combinationSum3(k, n):
        res = [[0, []]] # sum, combo
        for num in range(1, 10):
            res.extend([[num_sum + num, nums + [num]] for num_sum, nums in res if len(nums) < k])
        return [nums for num_sum, nums in res if num_sum == n and len(nums) == k]