class Answer(object):'''26. Remove Duplicates from Sorted Array'''
    def removeDuplicates(nums):
        i = 0
        nums.append('dummy')
        for j in range(1, len(nums)):
            if nums[j - 1] != nums[j]:
                nums[i] = nums[j - 1]
                i += 1
        while i != len(nums): nums.pop()