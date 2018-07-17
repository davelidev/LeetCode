class Answer(object):
'''124. Binary Tree Maximum Path Sum'''
    # Recursion, each returns 2 values, first one without going through the root, the second one goes through the root. take the max, and return max(left_no_root, right_no_root, root + left_root + right_root)
    def maxPathSum(root):
        def _maxPathSum(node):
            if not node: return float('-inf'), float('-inf')
            (l_i, l_x), (r_i, r_x) = _maxPathSum(node.left), _maxPathSum(node.right)
            inc = max(l_i, r_i, 0) + node.val
            exc = max(l_x, r_x, max(l_i, 0) + max(r_i, 0) + node.val)
            return inc, exc
        
        return max(_maxPathSum(root))