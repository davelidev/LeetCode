class Answer(object):
'''337. House Robber III'''
    def rob(root):
        def _rob(node):
            if not node: return 0, 0
            inc_l, not_inc_l = _rob(node.left)
            inc_r, not_inc_r = _rob(node.right)
            inc_node = node.val + not_inc_l + not_inc_r
            not_inc_node = max(inc_l, not_inc_l) + max(inc_r, not_inc_r)
            return inc_node, not_inc_node
        return max(_rob(root))