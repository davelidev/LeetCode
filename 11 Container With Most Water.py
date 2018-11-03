class Answer(object):'''11. Container With Most Water'''
    def maxArea(height):
        i, j = 0, len(height) - 1
        water = 0
        while i < j:
            h = min(height[i], height[j])
            water = max(h * (j - i), water)
            while i < j and height[j] <= h: j -= 1
            while i < j and height[i] <= h: i += 1
        return water