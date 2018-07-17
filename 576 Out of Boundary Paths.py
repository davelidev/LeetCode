class Answer(object):
'''576. Out of Boundary Paths'''
    def findPaths(m, n, N, i, j):
        diff = zip([-1, 0, 1, 0],
                   [0, -1, 0, 1])
        def move(bfs):
            from collections import defaultdict
            new_bfs = defaultdict(int)
            for (x, y, step), count in bfs.iteritems():
                for x_dif, y_dif in diff:
                    if 0 <= (x + x_dif) < m and 0 <= (y + y_dif) < n:
                        new_bfs[(x + x_dif, y + y_dif, step + 1)] += count
            return  new_bfs, sum(new_bfs.values())

        bfs = {(i, j, 0): 1}
        total = 0
        pre_count = 1
        while bfs and N > 0:
            bfs, count = move(bfs)
            total += pre_count * 4 - count
            pre_count = count
            N -= 1
        return total % (10**9 + 7)

    def findPaths(m, n, N, i, j):
        grid = [[0] * n for _ in range(m)]
        dif = zip([1, 0, -1, 0],
                  [0, 1, 0, -1])
        grid[i][j] = 1
        def sum_neighbor():
            new_grid = [[0] * n for _ in range(m)]
            for x in range(m):
                for y in range(n):
                    adjs = [(x + x_dir, y + y_dir) for x_dir, y_dir in dif]
                    new_grid[x][y] += sum(grid[a][b] for a, b in adjs if 0 <= a < m and 0 <= b < n)
            return new_grid
        prev_count = 1
        total_count = 0
        for _ in range(N):
            grid = sum_neighbor()
            cur_count = sum(map(sum, grid))
            total_count += prev_count * 4 - cur_count
            prev_count = cur_count
        return total_count %(10**9 + 7)

    def findPaths(m, n, N, i, j):
        grid = [[0] * n for _ in range(m)]
        dif = zip([0, 1, 0, -1],
                  [1, 0, -1, 0])
        for _ in range(N):
            grid = [[
                sum(grid[adj_x][adj_y] if (0 <= adj_x < m and 0 <= adj_y < n) else 1
                    for adj_x, adj_y in [(x + x_dir, y + y_dir) for x_dir, y_dir in dif])
                for y in range(n)]
                for x in range(m)]
        return grid[i][j] % (10**9 + 7)