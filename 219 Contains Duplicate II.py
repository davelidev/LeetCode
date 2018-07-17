class Answer(object):
'''219. Contains Duplicate II'''
    def containsNearbyDuplicate(nums, k):
        past = {}
        for i, num in enumerate(nums):
            if num in past and i - past[num] <= k: return True
            past[num] = i
        return False