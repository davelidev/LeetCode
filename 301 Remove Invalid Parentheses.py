class Answer(object):
'''301. Remove Invalid Parentheses'''
    def check_valid(lst):
        level = 0
        for item in lst:
            if level < 0: return False
            elif item == '(': level += 1
            elif item == ')': level -= 1
            elif level == 0: return Fale
        return level == 0
    lst = []
    for char in input_lst:
        if char == '(' or char == ')': lst.append(char)
        elif lst and lst[-1] != '(' and lst[-1] != ')': lst[-1] += char
        else: lst.append(char)
    bfs = [lst]
    bfs_secondary = []
    res = []
    while not res:
        while bfs:
            next_elem = bfs.pop()[:]
            if check_valid(next_elem): res.append(next_elem)
            elif not res:
                for i in range(len(next_elem)):
                    if next_elem[i] == '(' or next_elem[i] == ')':
                        bfs_secondary.append(next_elem[:i] + next_elem[i + 1:])
        bfs = bfs_secondary
        bfs_secondary = []