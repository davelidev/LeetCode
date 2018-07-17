class Answer(object):
'''217. Contains Duplicate'''
    def containsDuplicate(nums):
        return len(nums) != len(set(nums))