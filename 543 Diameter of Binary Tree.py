class Answer(object):
'''543. Diameter of Binary Tree'''
    def diameterOfBinaryTree(root):
        self.max = 0
        def _longest_len(node):
            if not node: return 0
            l, r = _longest_len(node.left), _longest_len(node.right)
            self.max = max(self.max, l + r)
            return max(l, r) + 1
        _longest_len(root)
        return self.max