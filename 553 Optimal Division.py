class Answer(object):
'''553. Optimal Division'''
    def optimalDivision(nums):
        return '/'.join(map(str, nums)) if len(nums) <= 2 else             '%d/(%s)' %(nums[0], '/'.join(map(str, nums[1:])))