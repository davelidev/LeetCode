class Answer(object):
'''435. Non-overlapping Intervals'''
    def eraseOverlapIntervals(intervals):
        def not_overlap(int1, int2):
            return min(int1.end, int2.end) <= max(int1.start, int2.start)
        def eft_cmp(x, y):
            if x.end < y.end or x.start < y.start:
                return -1
            elif x.end > y.end or x.start > y.start:
                return 1
            else:
                return 0
        intervals.sort(eft_cmp)
        count = 0
        prev = None
        for interval in intervals:
            if (prev and not_overlap(prev, interval)) or not prev :
                count += 1
                prev = interval
        return len(intervals) - count