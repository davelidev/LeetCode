class Answer(object):
'''515. Find Largest Value in Each Tree Row'''
    def largestValues(root):
        if not root: return []
        bfs = [root]
        res = []
        while bfs:
            res.append(max(map(lambda node: node.val, bfs)))
            bfs = [kid for node in bfs for kid in (node.left, node.right) if kid]
        return res