class Answer(object):
'''94. Binary Tree Inorder Traversal'''
    def inorderTraversal(root):
        stack = []
        def move_left(node):
            while node:
                stack.append(node)
                node = node.left
        move_left(root)
        res = []
        while stack:
            next_elem = stack.pop()
            res.append(next_elem.val)
            if next_elem.right: move_left(next_elem.right)
        return res