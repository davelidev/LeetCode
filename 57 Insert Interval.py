class Answer(object):'''57. Insert Interval'''
        def insert(intervals, newInterval):
        left, right = [], []
        s, e = newInterval.start, newInterval.end
        for interval in intervals:
            if interval.end < s: left.append(interval)
            elif e < interval.start: right.append(interval)
            else: s, e = min(interval.start, s), max(interval.end, e)
        return left + [Interval(s, e)] + right