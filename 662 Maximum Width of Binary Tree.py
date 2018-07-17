class Answer(object):
'''662. Maximum Width of Binary Tree'''
    def widthOfBinaryTree(root):
        if not root: return
        bfs = [(root, 1)]
        max_len = 0
        while bfs:
            max_len = max(max_len, bfs[-1][1] - bfs[0][1] + 1)
            bfs = [(kid, pos * 2 + (kid == node.right))
                   for node, pos in bfs
                   for kid in (node.left, node.right) if kid]
        return max_len