class Answer(object):
'''101. Symmetric Tree'''
    def isSymmetric(root):
        def _isSymmetric(t1, t2):
            if not t1 and not t2: return True
            elif (not t1 and t2) or (t1 and not t2): return False
            return t1.val == t2.val and                     _isSymmetric(t1.left, t2.right) and                     _isSymmetric(t1.right, t2.left)
        if not root: return True
        return _isSymmetric(root.left, root.right)