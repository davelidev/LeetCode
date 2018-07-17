class Answer(object):
'''236. Lowest Common Ancestor of a Binary Tree'''
    def lowestCommonAncestor(self, root, p, q):
        def _searchAncestor(node):
            if node in [p, q, None]: return node
            else:
                l, r = _searchAncestor(node.left), _searchAncestor(node.right)
                return node if (l and r) else (l or r)
        return _searchAncestor(root)