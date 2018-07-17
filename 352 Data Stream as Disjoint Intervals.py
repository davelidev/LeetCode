class Answer(object):
'''352. Data Stream as Disjoint Intervals'''
    # This is a variation of "128. Longest Consecutive Sequence". In addition, keep a list of start times to reconstruct at later time.
    class SummaryRanges(object):
        def __init__(self):
            self.start_times = set()
            self.start_time_to_num_of_elems = {}
        def addNum(self, num):
            consecutive = self.start_time_to_num_of_elems
            start_times = self.start_times
            if num not in consecutive:
                size = 1
                left = right = None
                if num - 1 in consecutive:
                    size += consecutive[num - 1]
                    left = (num - 1) - (consecutive[num - 1] - 1)
                if num + 1 in consecutive:
                    start_times.remove(num + 1)
                    size += consecutive[num + 1]
                    right = (num + 1) + (consecutive[num + 1] - 1)
                    consecutive[right] = size
                if left is not None:
                    consecutive[left] = size
                    start_times.add(left)
                else:
                    start_times.add(num)
                consecutive[num] = size
        def getIntervals(self):
            res = []
            for start_time in sorted(self.start_times):
                res.append([start_time, start_time + self.start_time_to_num_of_elems[start_time] - 1])
            return res if res else None