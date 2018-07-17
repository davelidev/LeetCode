class Answer(object):
'''227. Basic Calculator II'''
    def calculate(s):
        ops = {
            '+': lambda x, y : x + y,
            '-': lambda x, y : x - y,
            '*': lambda x, y : x * y,
            '/': lambda x, y : x / y
        }
        stack = []
        buf = []
        for char in s.replace(' ', ''):
            if char.isdigit():
                buf.append(char)
            else:
                stack.append(int(''.join(buf)))
                stack.append(char)
                buf = []
        stack.append(int(''.join(buf)))
        new_stack = []
        is_op = False
        for i, item in enumerate(stack):
            if not is_op and new_stack and new_stack[-1] in '*/':
                op, l, r = new_stack.pop(), new_stack.pop(), item
                new_stack.append(ops[op](l, r))
            else:
                new_stack.append(item)
            is_op = not is_op
        new_stack, stack, is_op = [], new_stack, False
        for i, item in enumerate(stack):
            if not is_op and new_stack and new_stack[-1] in '+-':
                op, l, r = new_stack.pop(), new_stack.pop(), item
                new_stack.append(ops[op](l, r))
            else:
                new_stack.append(item)
            is_op = not is_op
        return new_stack[0]