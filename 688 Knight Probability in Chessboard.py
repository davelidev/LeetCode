class Answer(object):
'''688. Knight Probability in Chessboard'''
    def knightProbability(N, K, r, c):
        x_y_diff = [2, -2, 1, -1] 
        diff = [(x, y) for x in x_y_diff for y in x_y_diff if abs(x) != abs(y)]
        from collections import defaultdict
        def move(bfs):
            new_bfs = defaultdict(int)
            for (x, y, step), count in bfs.iteritems():
                for x_dif, y_dif in diff:
                    if 0 <= (x + x_dif) < N and 0 <= (y + y_dif) < N:
                        new_bfs[(x + x_dif, y + y_dif, step + 1)] += count
            return  new_bfs, sum(new_bfs.values())
        bfs = {(r, c, 0): 1}
        iteration = total = 0
        pre_count = 1
        while bfs and iteration < K:
            bfs, count = move(bfs)
            total += pre_count * len(diff) - count
            pre_count = count
            iteration += 1
        return  float(sum(bfs.values())) / len(diff) ** K

    def knightProbability(N, K, r, c):
        x_y_diff = [2, -2, 1, -1] 
        diff = [(x, y) for x in x_y_diff for y in x_y_diff if abs(x) != abs(y)]
        board = [[0] * N for _ in range(N)]
        board[r][c] = 1
        for _ in range(K):
            board = [[sum((board[x + x_dif][y + y_dif]
                           for x_dif, y_dif in diff
                           if 0 <= (x + x_dif) < N and 0 <= (y + y_dif) < N))
                      for y in range(N)]
                     for x in range(N)]
        return float(sum(map(sum, board))) / (len(diff) ** K)