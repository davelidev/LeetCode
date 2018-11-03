class Answer(object):'''56. Merge Intervals'''
    def merge(intervals):
        res = []
        intervals = [[interval.start, interval.end] for interval in intervals]
        intervals.sort(reverse=True)
        while intervals:
            res.append(intervals.pop())
            if len(res) >= 2:
                [a,b], [c,d] = res[-2], res[-1]
                print [a,b], [c,d], a <= c <= b
                if a <= c <= b:
                    res.pop(); res.pop()
                    res.append([a, max(b, d)])
        return [Interval(start, end) for start, end in res]