class Answer(object):'''33. Search in Rotated Sorted Array'''
    def search(nums, target):
        lo, hi = 0, len(nums) - 1
        while lo <= hi:
            mid = (lo + hi) / 2
            if nums[mid] == target: return mid
            elif nums[lo] <= target < nums[mid] or                 (nums[lo] > nums[mid] and (target < nums[mid] or target >= nums[lo])): hi = mid - 1
            else: lo = mid + 1
        return -1

    def search( nums, target):
        if not nums: return -1
        # find lowest_idx
        lo, hi = 0, len(nums) - 1
        while lo < hi:
            mid = (lo + hi) / 2
            if nums[mid] < nums[hi]: hi = mid
            else: lo = mid + 1
        lo = lo or len(nums)
        lo, hi = (0, lo) if nums[0] <= target <= nums[lo - 1] else (lo, len(nums))
        from bisect import bisect_left
        l_idx = bisect_left(nums, target, lo, hi)
        return l_idx if l_idx < len(nums) and nums[l_idx] == target else -1