class Answer(object):
'''485. Max Consecutive Ones'''
    def findMaxConsecutiveOnes(nums):
        max_len = 0
        for j, val in enumerate(nums):
            if val:
                if j - 1 < 0 or not (nums[j - 1]): i = j
                max_len = max(max_len, j - i + 1)
        return max_len