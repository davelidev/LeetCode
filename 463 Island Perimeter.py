class Answer(object):
'''463. Island Perimeter'''
    def islandPerimeter(grid):
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]:
                    adjs = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                    for adj in adjs:
                        if not (0 <= adj[0] < len(grid) and 0 <= adj[1] < len(grid[0])) or                             not grid[adj[0]][adj[1]]:
                            count += 1
        return count