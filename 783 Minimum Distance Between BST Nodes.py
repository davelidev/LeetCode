class Answer(object):
'''783. Minimum Distance Between BST Nodes'''
    # iterative
    def minDiffInBST(root):
        # in order traversal
        cur = root
        stack = []
        def traverse_left(cur):
            while cur:
                stack.append(cur)
                cur = cur.left
        traverse_left(root)
        prev = None
        min_dif = float('inf')
        while stack:
            cur = stack[-1].val
            traverse_left(stack.pop().right)
            if prev is not None: min_dif = min(cur - prev, min_dif)
            prev = cur
        return min_dif

    # recursive
    def minDiffInBST(root):
        self.prev, self.min_dif = float('-inf'), float('inf')
        def in_order(node):
            if not node: return
            in_order(node.left)
            self.min_dif = min(self.min_dif, node.val - self.prev)
            self.prev = node.val
            in_order(node.right)
        in_order(root)
        return self.min_dif