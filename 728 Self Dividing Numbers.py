class Answer(object):
'''728. Self Dividing Numbers'''
    def selfDividingNumbers(left, right):
        res = []
        for i in range(left, right + 1):
            x = i
            is_div = True
            while is_div and x:
                x, mod = divmod(x, 10)
                if not mod or i % mod != 0: is_div = False
            if is_div: res.append(i)
        return res