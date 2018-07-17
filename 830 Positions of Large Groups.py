class Answer(object):
'''830. Positions of Large Groups'''
    def largeGroupPositions(S):
        stack = []
        prev = None
        for i, cur in enumerate(S):
            if cur != prev: stack.append(i)
            prev = cur
        stack.append(len(S))
        return [(stack[i - 1], stack[i] - 1) for i in range(1, len(stack)) if stack[i] - stack[i-1] >= 3]