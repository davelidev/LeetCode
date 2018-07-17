class Answer(object):
'''84. Largest Rectangle in Histogram'''
    def largestRectangleArea(heights):
        heights.append(0)
        dp = []
        max_area = 0
        for i, height in enumerate(heights):
            left = i
            while dp and dp[-1][1] > height:
                left = dp[-1][0]
                j, j_height = dp.pop()
                max_area = max(max_area, j_height * (i - j))
            dp.append((left, height))
        return max_area