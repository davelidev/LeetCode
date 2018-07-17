class Answer(object):
'''108. Convert Sorted Array to Binary Search Tree'''
    def sortedArrayToBST(nums):
        def _sortedArrayToBST(i, j):
            if i >= j: return
            mid = (i + j) / 2
            node = TreeNode(nums[mid])
            node.left = _sortedArrayToBST(i, mid)
            node.right = _sortedArrayToBST(mid + 1, j)
            return node

        return _sortedArrayToBST(0, len(nums))