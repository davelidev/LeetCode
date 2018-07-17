class Answer(object):
'''637. Average of Levels in Binary Tree'''
    def averageOfLevels(root):
        if not root: return []
        avgs = []
        bfs = [root]
        while bfs:
            vals = [node.val for node in bfs]
            avgs.append(float(sum(vals)) / len(vals))
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        return avgs