class Answer(object):
'''173. Binary Search Tree Iterator'''
    class BSTIterator(object):
        def __init__(self, root):
            cur = root
            self.stack = []
            while cur:
                self.stack.append(cur)
                cur = cur.left

        def hasNext(self):
            return bool(self.stack)

        def next(self):
            cur = ret = self.stack.pop()
            cur = cur.right
            while cur:
                self.stack.append(cur)
                cur = cur.left
            return ret.val