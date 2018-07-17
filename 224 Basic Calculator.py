class Answer(object):
'''224. Basic Calculator'''
    def calculate(s):
        s = list(('((%s)'%s).replace(' ', ''))
        nested_brackets, num = [], []
        for i, c in enumerate(s):
            if c == '(':
                nested_brackets.append([0, '+'])
                num = []
            elif c == '-' and s[i-1] in '+-(':
                prev_sign = nested_brackets[-1][1]
                nested_brackets[-1][1] = '+' if prev_sign == '-' else '+'
            elif c in '+-)':
                if num:
                    num = int(''.join(num))
                    sign = (1 if nested_brackets[-1][1] == '+' else -1)
                    nested_brackets[-1][0] += sign * num
                    num = []
                    if c == ')':
                        num = list(str(nested_brackets[-1][0]))
                        nested_brackets.pop()
                    else: nested_brackets[-1][1] = c
            elif c.isdigit(): num.append(c)
        return int(''.join(num))