class Answer(object):
'''145. Binary Tree Postorder Traversal'''
    def postorderTraversal(root):
        if not root: return []
        bfs = [root]
        while any(None if type(node) != TreeNode else node for node in bfs):
            bfs = [kid for node in bfs
                   for kid in ([node.left, node.right, node.val] if type(node) == TreeNode else [node]) if kid is not None]
        return bfs