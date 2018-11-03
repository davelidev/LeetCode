class Answer(object):'''35. Search Insert Position'''
    def searchInsert(nums, target):
        from bisect import bisect_left
        return bisect_left(nums, target)