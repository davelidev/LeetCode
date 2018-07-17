class Answer(object):
'''102. Binary Tree Level Order Traversal'''
    def levelOrder(self, root):
        if not root: return []
        bfs = [root]
        res = []
        while bfs:
            res.append([node.val for node in bfs])
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        return res