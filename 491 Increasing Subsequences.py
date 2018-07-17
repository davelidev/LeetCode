class Answer(object):
'''491. Increasing Subsequences'''
    def findSubsequences(nums):
        res = {()}
        for num in nums:
            res |= { ary + (num, ) for ary in res if not ary or ary[-1] <= num }
        return [x for x in res if len(x) >= 2]