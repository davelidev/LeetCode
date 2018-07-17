class Answer(object):
'''542. 01 Matrix'''
    def updateMatrix(self, matrix):
        def get_adj(i, j):
            return filter(lambda pos: 0 <= pos[0] < len(matrix) and 0 <= pos[1] < len(matrix[0]),
                          [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]])
        bfs = []
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j]:
                    set_none = True
                    for adj_i, adj_j in get_adj(i, j):
                        set_none &= matrix[adj_i][adj_j] == 1 or matrix[adj_i][adj_j] is None
                    if set_none:
                        matrix[i][j] = None
                    else:
                        bfs.append([i, j])
        bfs2 = []
        level = 1
        while bfs:
            level += 1
            while bfs:
                i, j = bfs.pop()
                for adj_i, adj_j in get_adj(i, j):
                    if matrix[adj_i][adj_j] is None:
                        matrix[adj_i][adj_j] = level
                        bfs2.append([adj_i, adj_j])
            bfs, bfs2 = bfs2, bfs
        return matrix