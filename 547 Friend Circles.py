class Answer(object):
'''547. Friend Circles'''
    def findCircleNum(M):
        visited = set()
        res = 0
        for node in range(len(M)):
            if node not in visited:
                res += 1
                bfs = [node]
                while bfs:
                    visited |= set(bfs)
                    bfs = [adj for i in bfs for adj, val in enumerate(M[i]) if val and (adj not in visited)]
        return res