class Answer(object):
'''658. Find K Closest Elements'''
    def findClosestElements(arr, k, x):
        from bisect import bisect_left
        pos = bisect_left(arr, x, 0, len(arr))
        if pos - 1 > 0 and arr[pos] != x: pos -= 1
        i, j = pos, pos + 1
        while j - i < k:
            if 0 < i and j < len(arr):
                if ((x - arr[i - 1]) <= (arr[j] - x)): i -= 1
                else: j += 1
            elif 0 < i: i -= 1
            else: j += 1
        return arr[i:j]