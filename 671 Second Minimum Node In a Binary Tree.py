class Answer(object):
'''671. Second Minimum Node In a Binary Tree'''
    def findSecondMinimumValue(root):
        def next_mins(node, root_val, res=[]):
            if not node: return res
            if node.val != root_val: res.append(node.val)
            else:
                next_mins(node.left, root_val, res)
                next_mins(node.right, root_val, res)
                return res
        if not root: return -1
        next_mins = next_mins(root, root.val)
        return min(next_mins) if next_mins else -1