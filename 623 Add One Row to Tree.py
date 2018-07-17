class Answer(object):
'''623. Add One Row to Tree'''
    def addOneRow(self, root, v, d):
        if d == 1:
            new_node = TreeNode(v)
            new_node.left = root
            return new_node
        bfs = [root]
        prev = None
        for i in range(d - 2):
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        for node in bfs:
            new_node = TreeNode(v)
            node.left, new_node.left = new_node, node.left
            new_node = TreeNode(v)
            node.right, new_node.right = new_node, node.right
        return root