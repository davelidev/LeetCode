class Answer(object):
'''675. Cut Off Trees for Golf Event'''
    # First iterate through the forest and store the values along with their associated positions. Sort the values and iterate through the list to cut off the trees one by one while accumulating the steps taken. Searching can be done using bfs with a visited algorithm.
    def cutOffTree(forest):
        m, n = len(forest), len(forest[0]) if forest else 0
        search = sorted([(forest[i][j], i, j)
                         for i in range(m)
                         for j in range(n)
                         if forest[i][j] > 1])

        xy_dir = zip([1, 0, -1, 0], [0, 1, 0, -1])
        def search_tree(x1, y1, x2, y2, height):
            visited = [([None] * n) for i in range(m)]
            def get_adjs(x, y):
                return [(a, b)
                        for a, b in [(x + x_d, y + y_d) for x_d, y_d in xy_dir]
                        if 0 <= a < m and 0 <= b < n                         and 0 < forest[a][b]]
            bfss = [[(x1, y1)], [(x2, y2)]]
            if (x1, y1) == (x2, y2): return 0
            visited[x1][y1] = 0
            visited[x2][y2] = 1
            step = 0
            while all(bfss):
                new_bfs = []
                bfs_i = step % 2
                for x, y in bfss[bfs_i]:
                    for a_x, a_y in get_adjs(x, y):
                        if visited[a_x][a_y] is None:
                            new_bfs.append((a_x, a_y))
                            visited[a_x][a_y] = bfs_i
                        elif visited[a_x][a_y] != bfs_i: return step + 1
                step += 1
                bfss[bfs_i] = new_bfs
        
        prev_x, prev_y = 0, 0
        total_steps = 0
        for tree in search:
            height, x, y = tree
            steps = search_tree(prev_x, prev_y, x, y, height)
            prev_x, prev_y = x, y
            if steps is None: return -1
            total_steps += steps
        return total_steps