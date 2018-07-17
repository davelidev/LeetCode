class Answer(object):
'''652. Find Duplicate Subtrees'''
    def findDuplicateSubtrees(root):
        def _findDuplicateSubtrees(node, hash_to_count_node):
            if not node: return
            l_res = _findDuplicateSubtrees(node.left, hash_to_count_node)
            r_res = _findDuplicateSubtrees(node.right, hash_to_count_node)
            serial = (node.val, l_res, r_res)
            hash_val = hash(str(serial))
            hash_to_count_node.setdefault(hash_val, [0, node])
            hash_to_count_node[hash_val][0] += 1
            return serial

        hash_to_count_node = {}
        _findDuplicateSubtrees(root, hash_to_count_node)
        return [node for i, node in hash_to_count_node.values() if i >= 2]