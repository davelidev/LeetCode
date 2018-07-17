class Answer(object):
'''437. Path Sum III'''
    def pathSum(root, sum):
        def _pathSum(node, target, sums_count={0: 1}, so_far=0):
            if not node: return 0
            so_far += node.val
            count = sums_count.get(so_far - target, 0)
            sums_count.setdefault(so_far, 0)
            sums_count[so_far] += 1
            count += _pathSum(node.left, target, sums_count, so_far)
            count += _pathSum(node.right, target, sums_count, so_far)
            sums_count[so_far] -= 1
            if so_far in sums_count and not sums_count[so_far]: del sums_count[so_far]
            return count
        return _pathSum(root, sum)