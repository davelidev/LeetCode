class Answer(object):
'''606. Construct String from Binary Tree'''
    def tree2str(t):
        def _tree2str(node):
            if not node: return ''
            left = _tree2str(node.left)
            right = _tree2str(node.right)
            if not left and not right: return str(node.val)
            elif not right: return '%d(%s)' %(node.val, left)
            else: return '%d(%s)(%s)' %(node.val, left, right)
        return _tree2str(t)