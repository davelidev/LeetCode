class Answer(object):
'''814. Binary Tree Pruning'''
    def pruneTree(root):
        def _pruneTree(node):
            if not node: return True
            l, r =  _pruneTree(node.left), _pruneTree(node.right)
            if l: node.left = None
            if r: node.right = None
            return not node.val and l and r
        _pruneTree(root)
        return root