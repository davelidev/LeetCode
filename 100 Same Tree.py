class Answer(object):
'''100. Same Tree'''
    def isSameTree(p, q):
        def _isSameTree(node1, node2):
            if not any([node1, node2]): return True
            elif not all([node1, node2]): return False
            return node1.val == node2.val and                     _isSameTree(node1.left, node2.left) and                     _isSameTree(node1.right, node2.right)
        return _isSameTree(p, q)