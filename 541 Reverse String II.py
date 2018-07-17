class Answer(object):
'''541. Reverse String II'''
    def reverseStr(s, k):
        res = []
        is_even = True
        for i in range(0, len(s), k):
            part = s[i:min(i+k, len(s))]
            if is_even: part = part[::-1]
            res.extend(part)
            is_even = not is_even
        return ''.join(res)