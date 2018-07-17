class Answer(object):
'''852. Peak Index in a Mountain Array'''
    def peakIndexInMountainArray(A):
        l, r = 0, len(A)
        while True:
            mid = (l + r) / 2
            if A[mid - 1] < A[mid] > A[mid + 1]:
                return mid
            elif A[mid - 1] < A[mid] < A[mid + 1]:
                l = mid
            else:
                r = mid