class Answer(object):
'''687. Longest Univalue Path'''
    def longestUnivaluePath(root):
        self.max_len = 0
        def _longestUnivaluePath(node):
            if not node: return 0
            left_child = right_child = 0
            left_len, right_len = _longestUnivaluePath(node.left), _longestUnivaluePath(node.right)
            if node and node.left and node.left.val == node.val: left_child = left_len + 1
            if node and node.right and node.right.val == node.val: right_child = right_len + 1
            self.max_len = max(self.max_len, left_child + right_child)
            return max(left_child, right_child)
        _longestUnivaluePath(root)
        return self.max_len