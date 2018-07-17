class Answer(object):
'''449. Serialize and Deserialize bst'''
    class Codec:
        def serialize(self, root):
            def _serialize(node):
                if not node: return
                return (node.val, _serialize(node.left), _serialize(node.right))
            return str(_serialize(root))

        def deserialize(self, data):
            def _deserialize(input_tuple):
                if not input_tuple: return
                node = TreeNode(input_tuple[0])
                node.left = _deserialize(input_tuple[1])
                node.right = _deserialize(input_tuple[2])
                return node
            return  _deserialize(eval(data))