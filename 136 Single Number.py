class Answer(object):
'''136. Single Number'''
    def singleNumber(nums): return reduce(lambda x, y: x ^ y, nums, 0)