class Answer(object):'''892. Surface Area of 3D Shapes'''
    def surfaceArea(grid):
        n = len(grid)
        def surface(n): return n * 6 - (n - 1) * 2 if n else 0
        total = sum(surface(grid[i][j]) for i in range(n) for j in range(n))
        adj_ver = sum(min(grid[i][j], grid[i][j + 1]) * 2 for i in range(n) for j in range(n - 1))
        adj_hor = sum(min(grid[i][j], grid[i + 1][j ]) * 2 for i in range(n - 1) for j in range(n))
        return total - adj_ver - adj_hor