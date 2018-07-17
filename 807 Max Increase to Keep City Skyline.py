class Answer(object):
'''807. Max Increase to Keep City Skyline'''

    def maxIncreaseKeepingSkyline(self, grid):
        m, n = len(grid), len(grid[0])
        hor_view, ver_view = [0] * m, [0] * n
        for i in range(m):
            for j in range(n):
                hor_view[i] = max(hor_view[i], grid[i][j])
                ver_view[j] = max(ver_view[j], grid[i][j])
        total = sum(
            min(hor_view[i], ver_view[j]) - grid[i][j]
            for i in range(m) for j in range(n)
        )
        return total