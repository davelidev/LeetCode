class Answer(object):'''34. Search for a Range'''
    def searchRange(nums, target):
        from bisect import bisect_left, bisect_right
        left_idx, right_idx = bisect_left(nums, target), bisect_right(nums, target) - 1
        if left_idx == -1 or left_idx >= len(nums) or nums[left_idx] != target: left_idx = -1
        if right_idx == -1 or right_idx >= len(nums) or nums[right_idx] != target: right_idx = -1
        return left_idx, right_idx

    def searchRange(nums, target):
        self.target = target
        self.low, self.high = -1, -1
        def _searchRange(low, high):
            if low >= high: return
            mid = (low + high) / 2
            if nums[mid] == self.target:
                if mid - 1 < 0 or nums[mid - 1] != self.target: self.low = mid
                else: _searchRange(low, mid)
                if mid + 1 >= len(nums) or nums[mid + 1] != self.target: self.high = mid
                else: _searchRange(mid + 1, high)
            elif nums[mid] > self.target: _searchRange(low, mid)
            elif nums[mid] < self.target: _searchRange(mid + 1, high)
        _searchRange(0, len(nums))
        return self.low, self.high