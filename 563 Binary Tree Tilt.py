class Answer(object):
'''563. Binary Tree Tilt'''
    def findTilt(root):
        self.tilt = 0
        def get_sum_update_tilt(node):
            if not node: return 0
            l, r = get_sum_update_tilt(node.left), get_sum_update_tilt(node.right)
            self.tilt += abs(l - r)
            return l + r + node.val
        get_sum_update_tilt(root)
        return self.tilt