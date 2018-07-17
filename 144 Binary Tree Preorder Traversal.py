class Answer(object):
'''144. Binary Tree Preorder Traversal'''
    def preorderTraversal(root):
        if not root: return []
        bfs = [root]
        def check_node(node):
            if type(node) == TreeNode: return [node.val, node.left, node.right]
            else: return [node]
        while any(type(node) == TreeNode for node in bfs):
            bfs = [kid for node in bfs for kid in check_node(node) if kid is not None]
        return bfs