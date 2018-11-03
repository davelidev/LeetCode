class Answer(object):'''844. Backspace String Compare'''
    def backspaceCompare(S, T):
        def process_str(string):
            res = []
            for c in string:
                if c != '#':
                    res.append(c)
                elif res:
                    res.pop()
            return res
        return process_str(S) == process_str(T)