class Answer(object):
'''513. Find Bottom Left Tree Value'''
    def findBottomLeftValue(root):
        prev = []
        bfs = [root]
        while bfs:
            prev = bfs
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        return prev[0].val

    def findBottomLeftValue(root):
        def _findBottomLeftValue(node, depth=0, res=[]):
            if not node: return res
            if depth >= len(res):
                res.append(node.val)
            _findBottomLeftValue(node.left, depth + 1, res)
            _findBottomLeftValue(node.right, depth + 1, res)
            return res
        return _findBottomLeftValue(root)[-1]