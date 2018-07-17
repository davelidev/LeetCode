class Answer(object):
'''539. Minimum Time Difference'''
    def findMinDifference(self, timePoints):
        timePoints = map(lambda x: [int(i) for i in x.split(':')], timePoints)
        timePoints = map(lambda x: x[0] * 60 + x[1], timePoints)
        min_in_a_day = 24*60
        hash_to_bucket = [False] * min_in_a_day
        for time in timePoints:
            if hash_to_bucket[time]:
                return 0
            hash_to_bucket[time] = True
        prev = None
        first = None
        min_diff = float('inf')
        for i, val in enumerate(hash_to_bucket):
            if val:
                if prev is not None:
                    min_diff = min(i - prev, min_diff)
                else:
                    first = i
                prev = i
        return min(min_in_a_day - (prev - first), min_diff)