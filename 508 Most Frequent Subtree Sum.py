class Answer(object):
'''508. Most Frequent Subtree Sum'''
    def findFrequentTreeSum(root):
        if not root: return []
        from collections import defaultdict
        val_to_freq = defaultdict(int)
        def _findFrequentTreeSum(node):
            if not node: return 0
            tree_sum = node.val + _findFrequentTreeSum(node.left) + _findFrequentTreeSum(node.right)
            val_to_freq[tree_sum] += 1
            return tree_sum
        _findFrequentTreeSum(root)
        max_freq = max(val_to_freq.values())
        return [val for val, freq in val_to_freq.iteritems() if max_freq == freq]