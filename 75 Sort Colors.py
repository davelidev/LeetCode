class Answer(object):
'''75. Sort Colors'''
    def sortColors(nums):
        if len(nums) <= 1:
            return
        def _sortColors(start, color):
            i, j = start, len(nums) - 1
            while i < j:
                if nums[j] != color:
                    j -= 1
                elif nums[i] == color:
                    i += 1
                else:
                    nums[i], nums[j] = nums[j], nums[i]
            return i
        
        i = _sortColors(0, 0)
        _sortColors(i + 1 if nums[i] == 0 else i, 1)