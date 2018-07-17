class Answer(object):
'''391. Perfect Rectangle'''
    # hash all the corners, check count to ensure all corners have 2/4 count, and the outer corners have exactly 1 count
    def isRectangleCover(rectangles):

        def get_corners(bottom_left, top_right):
            return [tuple(top_right), tuple(bottom_left),
                    (top_right[0], bottom_left[1]), (bottom_left[0], top_right[1])]

        def get_area(bottom_left, top_right):
            return (top_right[1] - bottom_left[1]) * (top_right[0] - bottom_left[0])

        from collections import defaultdict
        hashed_count = defaultdict(int)
        top_right, bottom_left = (float('-inf'), ), (float('inf'), )
        area_by_acc = 0

        for rect in rectangles:
            top_right = max(top_right, tuple(rect[2:]))
            bottom_left = min(bottom_left, tuple(rect[:2]))
            corners = get_corners(rect[:2], rect[2:])
            area_by_acc += get_area(rect[:2], rect[2:])
            for corner in corners: hashed_count[corner] += 1

        counts = hashed_count.values()
        outer_corners_by_1_count = set(corner for corner, count in hashed_count.iteritems() if count == 1)
        outer_corners_by_loop = set(get_corners(bottom_left, top_right))

        area_by_outer_corners = get_area(bottom_left, top_right)
    
        return len(outer_corners_by_1_count) == 4 and             outer_corners_by_1_count == outer_corners_by_loop and             area_by_acc == area_by_outer_corners and             next((False for count in counts if count % 2 != 0 and count != 1), True)