class Answer(object):
'''661. Image Smoother'''
    def imageSmoother(M):
        if not M or not any(M): return M
        res = [[0] * len(M[0]) for _ in range(len(M))]
        for i in range(len(res)):
            for j in range(len(res[0])):
                dif = [-1, 0, 1]
                adjs = [M[x][y]
                        for x, y in
                        [(i + dif_x, j + dif_y) for dif_x in dif for dif_y in dif]
                        if 0 <= x < len(res) and 0 <= y < len(res[0])]
                res[i][j] = sum(adjs) / len(adjs)
        return res