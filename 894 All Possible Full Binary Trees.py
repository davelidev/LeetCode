class Answer(object):'''894. All Possible Full Binary Trees'''
    def allPossibleFBT(self, N):
        def construct(n):
            if n == 1: return [TreeNode(0)]
            trees = []
            for k in range(1, n, 2):
                left, right = construct(k), construct(n - k - 1)
                for l in left:
                    for r in right:
                        node = TreeNode(0)
                        node.left, node.right = l, r
                        trees.append(node)
            return trees
        return construct(N)