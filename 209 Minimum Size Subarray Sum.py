class Answer(object):
'''209. Minimum Size Subarray Sum'''
    def minSubArrayLen(s, nums):
        min_len = total = start = 0
        for end, num in enumerate(nums):
            total += num
            while total >= s:
                min_len = min(end - start + 1, min_len or float('inf'))
                total -= nums[start]
                start += 1
        return min_len