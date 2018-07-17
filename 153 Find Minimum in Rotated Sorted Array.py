class Answer(object):
'''153. Find Minimum in Rotated Sorted Array'''
    def findMin(nums):
        low, high = 0, len(nums)
        while low < high:
            mid = (low + high) / 2
            if mid + 1 >= len(nums):
                return min(nums[-1], nums[0])
            elif nums[mid] > nums[mid + 1]:
                return nums[mid + 1]
            elif nums[mid] < nums[0]:
                high = mid
            elif nums[mid] > nums[0]:
                low = mid