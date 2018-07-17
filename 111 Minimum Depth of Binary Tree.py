class Answer(object):
'''111. Minimum Depth of Binary Tree'''
    def minDepth(root):
        if not root: return 0
        depth, bfs = 0, [root]
        while bfs:
            depth += 1
            if any(node for node in bfs if not node.left and not node.right): return depth
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]