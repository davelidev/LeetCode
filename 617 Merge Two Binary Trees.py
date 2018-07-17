class Answer(object):
'''617. Merge Two Binary Trees'''
    def mergeTrees(t1, t2):
        def _merge_nodes(node1, node2):
            if not (node1 and node2): return node1 or node2
            node1.val += node2.val
            node1.left = _merge_nodes(node1.left, node2.left)
            node1.right = _merge_nodes(node1.right, node2.right)
            return node1
        return _merge_nodes(t1, t2)
            
            501. Find Mode in Binary Search Tree
    def findMode(root):
        counts = {}
        if not root: return []
        bfs = [root]
        while bfs:
            for node in bfs:
                counts[node.val] = counts.get(node.val, 0) + 1
            bfs = [kid for node in bfs for kid in [node.left, node.right] if kid]
        max_freq = max(counts.values())
        return [key for key, freq in counts.iteritems() if freq == max_freq]