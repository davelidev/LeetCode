class Answer(object):
'''669. Trim a Binary Search Tree'''
    def trimBST(root, L, R):
        def _trimBST(node, L, R):
            if not node: return
            elif node.val < L: return _trimBST(node.right, L, R)
            elif node.val > R: return _trimBST(node.left, L, R)
            else:
                node.left = _trimBST(node.left, L, R)
                node.right = _trimBST(node.right, L, R)
                return node
        return _trimBST(root, L, R)