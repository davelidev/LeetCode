class Answer(object):
'''404. Sum of Left Leaves'''
    def sumOfLeftLeaves(root):
        if not root: return 0
        left_leave_sum = 0
        bfs = [root]
        while bfs:
            left_leave_sum += sum(node.left.val for node in bfs if node.left and not node.left.right and not node.left.left)
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        return left_leave_sum