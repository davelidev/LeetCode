class Answer(object):
'''104. Maximum Depth of Binary Tree'''
    def maxDepth(root):
        if not root: return 0
        depth = 0
        bfs = [root]
        while bfs:
            depth += 1
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        return depth