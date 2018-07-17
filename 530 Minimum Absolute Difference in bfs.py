class Answer(object):
'''530. Minimum Absolute Difference in bfs'''
    def getMinimumDifference(root):
        if not root: return
        bfs = [root]
        vals = []
        while bfs:
            vals.extend([node.val for node in bfs])
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        vals.sort()
        return min(vals[i] - vals[i - 1] for i in range(1, len(vals)))