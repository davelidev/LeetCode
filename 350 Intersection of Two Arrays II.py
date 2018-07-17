class Answer(object):
'''350. Intersection of Two Arrays II'''
    def intersect(nums1, nums2):
        from collections import Counter
        counts1 = Counter(nums1)
        res = []
        for num in nums2:
            if counts1.get(num, 0):
                res.append(num)
                counts1[num] -= 1
        return res