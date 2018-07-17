class Answer(object):
'''747. Largest Number At Least Twice of Others'''
    def dominantIndex(nums):
        max_num, max_idx = max((val, idx) for idx, val in enumerate(nums))
        return max_idx if all(max_num >= num * 2 for num in nums if num != max_num) else -1