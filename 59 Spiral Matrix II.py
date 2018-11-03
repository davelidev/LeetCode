class Answer(object):'''59. Spiral Matrix II'''
    def generateMatrix(n):
        res = [[None] * n for _ in range(n)]
        cur_dir = x = 0
        y = -1
        new_pos_lambda = {
            0: lambda x, y: (x, y + 1),
            1: lambda x, y: (x + 1, y),
            2: lambda x, y: (x, y - 1),
            3: lambda x, y: (x - 1, y)
        }
        for i in range(1, n * n + 1):
            new_x, new_y = new_pos_lambda[cur_dir % 4](x, y)
            if not (0 <= new_x < n and  0 <= new_y < n) or res[new_x][new_y] is not None:
                cur_dir += 1
                new_x, new_y = new_pos_lambda[cur_dir % 4](x, y)
            res[new_x][new_y] = i
            x, y = new_x, new_y
        return res