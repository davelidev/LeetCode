class Answer(object):
'''283. Move Zeroes'''
    def moveZeroes(nums):
        i = 0
        for j, num in enumerate(nums):
            if num != 0:
                nums[i] = num
                i += 1
        for j in range(i, len(nums)):
            nums[j] = 0