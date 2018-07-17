class Answer(object):
'''496. Next Greater Element I'''
    def nextGreaterElement(findNums, nums):
        return [next((num for num in nums[nums.index(num_f):] if num > num_f), -1) for num_f in findNums]