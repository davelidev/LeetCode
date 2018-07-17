class Answer(object):
'''199. Binary Tree Right Side View'''
    def rightSideView(root):
        if not root: return []
        bfs = [root]
        res = []
        while bfs:
            res.append(bfs[-1].val)
            bfs = [kid for node in bfs for kid in (node.left, node.right) if kid]
        return res