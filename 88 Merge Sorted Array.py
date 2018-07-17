class Answer(object):
'''88. Merge Sorted Array'''
    def merge(self, nums1, m, nums2, n):        i, j = m - 1, n - 1
        for k in range(i + j + 1, -1, -1):
            if j < 0: nums1[k], i = nums1[i], i - 1
            elif i < 0: nums1[k], j = nums2[j], j - 1
            elif nums1[i] > nums2[j]: nums1[k], i = nums1[i], i - 1
            else: nums1[k], j = nums2[j], j - 1