class Answer(object):
'''554. Brick Wall'''
    def leastBricks(wall):
        split_count = {}
        split_max = 0
        split_idx = 0
        for row in wall:
            cur_sum = 0
            for i in range(0, len(row) - 1):
                cur_sum += row[i]
                split_count.setdefault(cur_sum, 0)
                split_count[cur_sum] += 1
                split_max = max(split_count[cur_sum], split_max)
        return len(wall) - split_max