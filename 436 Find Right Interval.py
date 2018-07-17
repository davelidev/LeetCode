class Answer(object):
'''436. Find Right Interval'''
    def findRightInterval(intervals):
        from bisect import bisect_left
        start_idx = sorted([i.start, idx] for idx, i in enumerate(intervals)) + [[float('inf'), -1]]
        return [start_idx[bisect_left(start_idx, [i.end])][1] for i in intervals]