class Answer(object):
'''331. Verify Preorder Serialization of a Binary Tree'''
    def isValidSerialization(self, preorder):
        preorder = preorder.split(',')
        diff = 1
        for node in preorder:
            diff -= 1
            if diff < 0: return False
            if node != '#': diff += 2
        return diff == 0

    def isValidSerialization(preorder):
        preorder = preorder.split(',')
        self.idx = 0
        def _isValidSerialization():
            if self.idx >= len(preorder): return False
            self.idx += 1
            if preorder[self.idx - 1] == '#': return True
            return _isValidSerialization() and _isValidSerialization()
        return _isValidSerialization() and self.idx == len(preorder)