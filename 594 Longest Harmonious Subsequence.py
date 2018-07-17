class Answer(object):
'''594. Longest Harmonious Subsequence'''
    def findLHS(nums):
        from collections import Counter
        counts = Counter(nums)
        return max([counts[x] + counts[x + 1] for x in counts if x + 1 in counts] or [0])