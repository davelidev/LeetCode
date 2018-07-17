class Answer(object):
'''162. Find Peak Element'''
    def findPeakElement(nums):
        if not nums: return 0
        def get_val(idx):
            return nums[idx] if 0 <= idx < len(nums) else float('-inf')
        lo, hi = 0, len(nums) - 1
        while True:
            mid = (lo + hi) / 2
            if get_val(mid - 1) < get_val(mid) > get_val(mid + 1): return mid
            elif get_val(mid - 1) > get_val(mid): hi = mid - 1
            else: lo = mid + 1