class Answer(object):
'''200. Number of Islands'''
    def numIslands(grid):
        if not grid or not grid[0]: return 0
        def convert(from_sym, to_sym, i, j):
            def get_adj(i, j):
                return [(x, y) for x, y in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                        if 0 <= x < len(grid) and 0 <= y < len(grid[0])]
            if grid[i][j] == from_sym:
                grid[i][j] = to_sym
                for adj in get_adj(i, j): convert(from_sym, to_sym, adj[0], adj[1])
        res = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    res += 1
                    convert('1', None, i, j)
        return res