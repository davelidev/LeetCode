class Answer(object):
'''697. Degree of an Array'''
    def findShortestSubArray(nums):
        from collections import Counter
        counts = Counter(nums)
        max_freq = max(counts.values())
        num_to_start_end = {key:(len(nums), 0) for key, val in counts.iteritems() if val == max_freq}
        for i, num in enumerate(nums):
            if num in num_to_start_end:
                start, end = num_to_start_end[num]
                num_to_start_end[num] = (min(i, start), max(i, end))
        return min([end - start + 1 for start, end in num_to_start_end.values()])