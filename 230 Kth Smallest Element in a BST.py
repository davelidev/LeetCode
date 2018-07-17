class Answer(object):
'''230. Kth Smallest Element in a BST'''
    def kthSmallest(root, k):
        stack = []
        def move_left(node):
            while node:
                stack.append(node)
                node = node.left
        move_left(root)
        for i in range(k):
            next_elem = stack.pop()
            if next_elem.right:
                move_left(next_elem.right)
        return next_elem.val