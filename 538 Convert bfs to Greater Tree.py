class Answer(object):
'''538. Convert bfs to Greater Tree'''
    def convertBST(root):
        self.cur_sum = 0
        def _convertBST(node):
            if not node: return
            _convertBST(node.right)
            node.val = self.cur_sum = node.val + self.cur_sum
            _convertBST(node.left)
        _convertBST(root)
        return root