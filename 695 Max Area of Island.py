class Answer(object):
'''695. Max Area of Island'''
    def maxAreaOfIsland(grid):
        def flip(i, j):
            if not grid[i][j]: return 0
            count = 0
            to_visit = {(i, j)}
            while to_visit:
                count += len(to_visit)
                for i, j in to_visit: grid[i][j] = None
                to_visit = set((x, y) for i, j in to_visit
                                      for x, y in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                                      if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y])
            return count
        
        max_area = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                max_area = max(max_area, flip(i, j))
        return max_area