class Answer(object):
'''598. Range Addition II'''
    def maxCount(m, n, ops):
        max_x = min([op[0] for op in ops if op[0]] or [0])
        max_y = min([op[1] for op in ops if op[1]] or [0])
        return (max_x * max_y) or (m * n)