class Answer(object):
'''640. Solve the Equation'''
    def solveEquation(equation):
        import re
        def count_x(s):
            return sum(int(num_x)
                       for num_x in re.findall('[\+-]\d*(?=x)',
                                               re.sub('(?<=[\+-])(?=x)', '1', s)))
        def count_num(s): return sum(int(num) for num in re.findall('[\+-]\d+(?=[+\-])', s + '+'))
        left, right = [(x if x.startswith('-') else '+' + x) for x in equation.split('=')]
        num_x = count_x(left) - count_x(right)
        val = count_num(right) - count_num(left)
        if not num_x and not val: return "Infinite solutions"
        elif not num_x: return 'No solution'
        return 'x=%d' % (val / num_x)