class Answer(object):
'''106. Construct Binary Tree from Inorder and Postorder Traversal'''
    def buildTree(inorder, postorder):
        def _buildTree(i_l, i_r):
            if i_l >= i_r: return
            node = TreeNode(postorder.pop())
            elem_idx = inorder.index(node.val)
            node.right = _buildTree(elem_idx + 1, i_r)
            node.left = _buildTree(i_l, elem_idx)
            return node
        return _buildTree(0, len(inorder))