class Answer(object):
'''149. Max Points on a Line'''
    def maxPoints(self, points):
    ''' O(V**2) time and O(V) space
        for each node, calculate the slope with adjacent nodes and keep track of the max
        edge cases => same coordinates, infinite slope, convert to float'''
        import numpy as np
        max_count = 0
        for i, point1 in enumerate(points):
            x1, y1 = point1.x, point1.y
            slope_cnt = {}
            same = 0
            for j, point2 in enumerate(points):
                if i != j:
                    x2, y2 = point2.x, point2.y
                    if (x1, y1) != (x2, y2):
                        slope = np.longdouble(y2 - y1) / (x2 - x1) if (x2 != x1) else 'inf'
                        slope_cnt[slope] = slope_cnt.get(slope, 1) + 1
                    else:
                        same += 1
            max_count = max(max(slope_cnt.values() or [1]) + same, max_count)
        return max_count or int(bool(points))