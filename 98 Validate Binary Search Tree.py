class Answer(object):
'''98. Validate Binary Search Tree'''
    def isValidBST(root):
        def _isValidBST(node, min_val, max_val):
            if not node: return True
            if min_val < node.val < max_val and _isValidBST(node.left, min_val, node.val) and _isValidBST(node.right, node.val, max_val):
                return True
            return False
        return _isValidBST(root, float('-inf'), float('inf'))