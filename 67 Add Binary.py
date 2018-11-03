class Answer(object):'''67. Add Binary'''
    def addBinary(a, b):
        a, b, carry = [int(i) for i in list(a)], [int(i) for i in list(b)], 0
        res = []
        while carry or a or b:
            cur = (a.pop() if a else 0) + (b.pop() if b else 0) + carry
            carry, digit = cur / 2, cur % 2
            res.append(str(digit))
        return ''.join(list(reversed(res)))