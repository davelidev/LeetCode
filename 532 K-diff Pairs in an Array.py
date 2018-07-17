class Answer(object):
'''532. K-diff Pairs in an Array'''
    def findPairs(nums, k):
        from collections import Counter
        counts = Counter(nums)
        res = 0
        for val in counts:
            if k > 0 and val + k in counts or                 not k and counts[val] > 1:
                res += 1
        return res