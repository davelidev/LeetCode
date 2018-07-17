class Answer(object):
'''114. Flatten Binary Tree to Linked List'''
    def flatten(root):
        def _flatten(node):
            if not node: return
            flatten_left = _flatten(node.left)
            l_end = None
            if flatten_left:
                l_start, l_end = flatten_left
                node.right, l_end.right, node.left = l_start, node.right, None
                flatten_right = _flatten(l_end.right)
            else:
                flatten_right = _flatten(node.right)
            r_start, r_end = flatten_right if flatten_right else [None, None]
            
            if r_end:
                return node, r_end
            elif l_end:
                return node, l_end
            else:
                return node, node
        _flatten(root)