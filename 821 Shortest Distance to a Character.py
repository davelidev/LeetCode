class Answer(object):
'''821. Shortest Distance to a Character'''
    def shortestToChar(S, C):
        res = []
        last_C = float('-inf')
        for i, char in enumerate(S):
            if char == C:
                last_C = i
            res.append(i - last_C)
        last_C = float('inf')
        for i in range(len(S) -1, -1, -1):
            char = S[i]
            if char == C:
                last_C = i
            res[i] = min(res[i], last_C - i)
        return res