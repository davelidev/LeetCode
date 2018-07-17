class Answer(object):
'''686. Repeated String Match'''
    def repeatedStringMatch(A, B):
        for i in range(3):
            times = len(B) / len(A) + i
            if B in A * times:
                return times
        return -1
        112. Path Sum
    def hasPathSum(root, total):
        cur_path = []
        def _pathSum(node, sum_from_root):
            if not node:
                return False
            sum_from_root += node.val
            cur_path.append(node.val)
            if sum_from_root == total and not node.left and not node.right:
                return True
            res = _pathSum(node.left, sum_from_root) or _pathSum(node.right, sum_from_root)
            cur_path.pop()
            return res
        return _pathSum(root, 0)