class Answer(object):
'''278. First Bad Version'''
    def firstBadVersion(n):
        low, high = 1, n
        
        while low <= high:
            mid = (low + high) / 2
            print low, high, mid
            if isBadVersion(mid):
                if (mid - 1) < low or not isBadVersion(mid - 1): return mid
                high = mid - 1
            else:
                low = mid + 1