class Answer(object):
'''215. Kth Largest Element in an Array'''
    def findKthLargest(nums, k):
        return sorted(nums)[len(nums) - k]