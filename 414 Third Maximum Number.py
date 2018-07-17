class Answer(object):
'''414. Third Maximum Number'''
    def thirdMax(nums):
        if not nums: return
        max_three = []
        for num in set(nums):
            max_three.append(num)
            if len(max_three) > 3:
                max_three.remove(min(max_three))
        return min(max_three) if len(max_three) >= 3 else max(max_three)