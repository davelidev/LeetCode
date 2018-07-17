class Answer(object):
'''279. Perfect Squares'''
    def numSquares(self, n):
        bfs = [n]
        count = 0
        visited = {n}
        sq = {i: i**2 for i in range(1, int(n ** (1./2)) + 1)}
        while bfs:
            if not all(bfs): return count
            bfs = [(i - sq[j])
                      for i in bfs
                      for j in range(1, int(i ** (1./2)) + 1)
                      if (i - sq[j]) not in visited and (i - sq[j]) >= 0 and (visited.add(i - sq[j]) is None)]
            count += 1
        return 0