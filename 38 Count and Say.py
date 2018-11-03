class Answer(object):'''38. Count and Say'''
    def countAndSay(n):
        cur = ['1']
        for i in range(n - 1):
            stack = []
            for elem in cur:
                if not stack or stack[-1][1] != elem: stack.append([1, elem])
                else: stack[-1][0] += 1
            cur = [str(i) for pair in stack for i in pair]
        return ''.join(cur)