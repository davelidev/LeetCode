class Answer(object):
'''103. Binary Tree Zigzag Level Order Traversal'''
    def zigzagLevelOrder(root):
        if not root: return []
        bfs = [root]
        res = []
        while bfs:
            new_item = [node.val for node in bfs]
            res.append(new_item[::-1] if len(res) % 2 else new_item)
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        return res