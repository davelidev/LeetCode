class Answer(object):
'''107. Binary Tree Level Order Traversal II'''
    def levelOrderBottom(root):
        if not root: return []
        lvl_tra = []
        bfs = [root]
        while bfs:
            lvl_tra.append([node.val for node in bfs])
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        return lvl_tra[::-1]