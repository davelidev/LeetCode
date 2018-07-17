class Answer(object):
'''385. Mini Parser'''
    def deserialize(s):
        i = 0
        stack = [NestedInteger()]
        while i < len(s):
            if s[i] == '[':
                ni = NestedInteger()
                stack[-1].add(ni)
                stack.append(ni)
            elif s[i].isdigit() or s[i] == '-':
                j = i
                while j < len(s) and (s[j] not in ',]'): j += 1
                ni = NestedInteger(int(s[i:j]))
                stack[-1].add(ni)
                i = j - 1
            elif s[i] == ']':
                stack.pop()
            i += 1
        return  stack[0].getList()[0]