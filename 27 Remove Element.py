class Answer(object):'''27. Remove Element'''
    def removeElement(nums, val):
            i = 0
            for j, num in enumerate(nums):
                if val != num:
                    nums[i] = nums[j]
                    i += 1
            while i != len(nums): nums.pop()
            return i