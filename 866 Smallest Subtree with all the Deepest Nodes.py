class Answer(object):
'''866. Smallest Subtree with all the Deepest Nodes'''
    def subtreeWithAllDeepest(root):
        bfs = {root}
        while bfs:
            prev = bfs
            bfs = {kid for node in bfs for kid in [node.left, node.right] if kid}
        deepest = prev
        def dfs_deepest(node):
            if not node or node in deepest: return node
            l, r = dfs_deepest(node.left), dfs_deepest(node.right)
            return node if l and r else l or r
        return dfs_deepest(root)