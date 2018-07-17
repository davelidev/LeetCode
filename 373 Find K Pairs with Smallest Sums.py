class Answer(object):
'''373. Find K Pairs with Smallest Sums'''
    def kSmallestPairs(nums1, nums2, k):
        return [item[1:] for item in sorted([(sum([num1, num2]), num1, num2) for num1 in nums1[:k] for num2 in nums2[:k]])[:k]]