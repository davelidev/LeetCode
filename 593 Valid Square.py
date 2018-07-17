class Answer(object):
'''593. Valid Square'''
    def validSquare(p1, p2, p3, p4):
        points = [p1, p2, p3, p4]
        dists = [ ((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
                 for i in range(len(points)) for j in range(i + 1, len(points))]
        from collections import Counter
        dists = Counter(dists)
        keys = dists.keys()
        return len(keys) == 2 and             (dists[keys[0]] == 2 or dists[keys[0]] == 4) and             (dists[keys[1]] == 2 or dists[keys[1]] == 4)