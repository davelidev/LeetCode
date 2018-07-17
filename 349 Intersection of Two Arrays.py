class Answer(object):
'''349. Intersection of Two Arrays'''
    def intersection(self, nums1, nums2): return list(set(nums1) & set(nums2))