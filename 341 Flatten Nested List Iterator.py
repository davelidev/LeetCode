class Answer(object):
'''341. Flatten Nested List Iterator'''
    class NestedIterator(object):

        def __init__(self, nestedList):
            self.stack, self.cache = [[nestedList, 0]], None

        def next(self):
            self.hasNext()
            res, self.cache = self.cache, None
            return res
        
        def hasNext(self):
            if self.cache is not None: return True
            elif not self.stack: return False
            next_lst, next_idx = self.stack[-1]
            if next_idx < len(next_lst):
                if next_lst[next_idx].isInteger():
                    self.cache = next_lst[next_idx].getInteger()
                    self.stack[-1][1] += 1
                    return True
                else:
                    self.stack[-1][1] += 1
                    self.stack.append([next_lst[next_idx].getList(), 0])
                    return self.hasNext()
            else:
                self.stack.pop()
                return self.hasNext()