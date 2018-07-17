class Answer(object):
'''235. Lowest Common Ancestor of a Binary Search Tree'''
    def lowestCommonAncestor(root, p, q):
        def _lowestCommonAncestor(node):
            if not node or node == p or node == q: return node
            left, right = _lowestCommonAncestor(node.left), _lowestCommonAncestor(node.right)
            return node if (left and right) else (left or right)
        return _lowestCommonAncestor(root)