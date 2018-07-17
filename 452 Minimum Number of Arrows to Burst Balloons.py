class Answer(object):
'''452. Minimum Number of Arrows to Burst Balloons'''
    def findMinArrowShots(points):
        points.sort()
        start = None
        res = 0
        while points:
            last = points.pop()
            if start is None or not (last[0] <= start <= last[1]):
                start = last[0]
                res += 1
        return res