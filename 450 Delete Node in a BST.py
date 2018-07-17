class Answer(object):
'''450. Delete Node in a BST'''
        def _findMinNode(node):
            node = node.right
            while node.left:
                node = node.left
            return node
        def _deleteNode(root, key):
            if not root:
                return
            print root.val
            if root.val > key:
                root.left = _deleteNode(root.left, key)
            elif root.val < key:
                root.right = _deleteNode(root.right, key)
            else:
                if not (root.left and root.right):
                    if root.left:
                        return root.left
                    elif root.right:
                        return root.right
                    else:
                        return
                else:
                    min_node = _findMinNode(root)
                    root.val = min_node.val
                    root.right = _deleteNode(root.right, min_node.val)
            return root