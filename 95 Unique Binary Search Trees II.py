class Answer(object):
'''95. Unique Binary Search Trees II'''
    def generateTrees(n):
        lst = range(1, n + 1)
        if not n: return []
        def clone(node):
            if not node: return
            new_node = TreeNode(node.val)
            new_node.left = clone(node.left)
            new_node.right = clone(node.right)
            return new_node
        def _generateTrees(i, j):
            if i >= j: return [None]
            res = []
            for k in range(i, j):
                left = _generateTrees(i, k)
                right = _generateTrees(k + 1, j)
                for l in left:
                    for r in right:
                        node = TreeNode(lst[k])
                        node.left = clone(l)
                        node.right = clone(r)
                        res.append(node)
            return res
        return _generateTrees(0, len(lst))