class Answer(object):
'''150. Evaluate Reverse Polish Notation'''
    def evalRPN(tokens):
        stack = []
        str_to_expr = {
            '+':lambda x, y: x + y,
            '-':lambda x, y: x - y,
            '*':lambda x, y: x * y,
            '/':lambda x, y: x / y
        }
        for char in tokens:
            if char in '+-*/':
                a, b = stack.pop(), stack.pop()
                stack.append(str_to_expr[char](b, a))
            else:
                stack.append(int(char))
        return stack[0]