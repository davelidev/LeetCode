class Answer(object):
'''113. Path Sum II'''
    def pathSum(root, total):
        cur_path = []
        res = []
        def _pathSum(node, sum_from_root):
            if not node:
                return
            sum_from_root += node.val
            cur_path.append(node.val)
            if sum_from_root == total and not node.left and not node.right:
                res.append(cur_path[:])
            _pathSum(node.left, sum_from_root)
            _pathSum(node.right, sum_from_root)
            cur_path.pop()
        _pathSum(root, 0)
        return res