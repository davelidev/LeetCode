class Answer(object):
'''560. Subarray Sum Equals K'''
    def subarraySum(nums, k):
        sum_from_left = 0
        sum_count = {0:1}
        res = 0
        for i, num in enumerate(nums):
            sum_from_left += num
            target_key = sum_from_left - k
            if target_key in sum_count:
                res += sum_count[target_key]
            sum_count.setdefault(sum_from_left, 0)
            sum_count[sum_from_left] += 1
        return res