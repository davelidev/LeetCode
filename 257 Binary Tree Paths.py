class Answer(object):
'''257. Binary Tree Paths'''
    def binaryTreePaths(root):
        if not root: return []
        bfs = [(root, str(root.val))]
        paths = []
        while bfs:
            paths.extend([path for node, path in bfs if not node.left and not node.right])
            bfs = [(kid, "%s->%s" %(path, str(kid.val))) for node, path in bfs for kid in [node.left, node.right] if kid]
        return paths