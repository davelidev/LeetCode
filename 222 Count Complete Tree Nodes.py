class Answer(object):
'''222. Count Complete Tree Nodes'''
    def countNodes(root):
        def _countNodes(node):
            left_node, left_depth = node, 0
            while left_node: left_node, left_depth = left_node.left, left_depth + 1
            right_node, right_depth = node, 0
            while right_node: right_node, right_depth = right_node.right, right_depth + 1
            if left_depth == right_depth: return 2 ** left_depth - 1
            else: return _countNodes(node.left) + 1 + _countNodes(node.right)
        return _countNodes(root)