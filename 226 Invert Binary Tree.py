class Answer(object):
'''226. Invert Binary Tree'''
    def invertTree(root):
        if not root: return
        bfs = [root]
        while bfs:
            for node in bfs: node.left, node.right = node.right, node.left
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        return root