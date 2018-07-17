class Answer(object):
'''733. Flood Fill'''
    def floodFill(image, sr, sc, newColor):
        stack = [(sr, sc)]
        old_col, new_col = image[sr][sc], newColor
        if old_col == new_col: return image
        while stack:
            i, j = stack.pop()
            image[i][j] = new_col
            stack.extend([(x, y) for x, y in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]                         if 0 <= x < len(image) and 0 <= y < len(image[0]) and image[x][y] == old_col])
        return image