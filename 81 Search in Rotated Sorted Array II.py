class Answer(object):
'''81. Search in Rotated Sorted Array II'''
    def search(nums, target):
        if not nums: return False
        lo, hi = 0, len(nums) - 1
        while lo <= hi:
            while (hi - lo) >= 1 and nums[hi] == nums[hi - 1]: hi -= 1
            mid = (lo + hi) / 2
            if nums[mid] == target: return True
            elif nums[lo] <= target < nums[mid] or                 (nums[lo] > nums[mid] and (target < nums[mid] or target >= nums[lo])): hi = mid - 1
            else: lo = mid + 1
        return False