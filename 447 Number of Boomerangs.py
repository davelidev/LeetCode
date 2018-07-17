class Answer(object):
'''447. Number of Boomerangs'''
    def numberOfBoomerangs(points):
        res = 0
        for i, [x, y] in enumerate(points):
            dist_to_point = {}
            for j, [adj_x, adj_y] in enumerate(points):
                if i != j:
                    key = (x - adj_x) ** 2 + (y - adj_y) ** 2
                    dist_to_point[key] = dist_to_point.setdefault(key, 0) + 1
            res += sum([val * (val - 1) for val in dist_to_point.values()])
        return res