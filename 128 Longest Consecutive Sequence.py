class Answer(object):
'''128. Longest Consecutive Sequence'''
    def longestConsecutive(nums):
        consecutive = {}
        max_size = 0
        for num in nums:
            if num not in consecutive:
                size = 1
                left = right = None
                if num - 1 in consecutive:
                    size += consecutive[num - 1]
                    left = (num - 1)-(consecutive[num - 1] - 1)
                if num + 1 in consecutive:
                    size += consecutive[num + 1]
                    right = (num + 1) + (consecutive[num + 1] - 1)
                    consecutive[right] = size
                if left is not None:
                    consecutive[left] = size
                consecutive[num] = size
                max_size = max(max_size, size)
        return max_size