class Answer(object):
'''682. Baseball Game'''
    def calPoints(ops):
        stack = []
        for item in ops:
            if item == 'D': stack.append(stack[-1] * 2)
            elif item == '+': stack.append(stack[-1] + stack[-2])
            elif item == 'C': stack.pop()
            else: stack.append(int(item))
        return sum(stack)