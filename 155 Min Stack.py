class Answer(object):
'''155. Min Stack'''
    class MinStack(object):

        def __init__(self):
            self.lst = []

        def push(self, x):
            cur_min = self.lst[-1][0] if self.lst else float('inf')
            self.lst.append([min(x, cur_min), x])

        def pop(self):
            self.lst.pop()

        def top(self):
            return self.lst[-1][1]

        def getMin(self):
            return self.lst[-1][0]