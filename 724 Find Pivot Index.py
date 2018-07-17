class Answer(object):
'''724. Find Pivot Index'''
    def pivotIndex(nums):
        right_sums = nums + [0]
        for i in range(len(right_sums) - 2, -1, -1): right_sums[i] += right_sums[i + 1]
        left_sum = 0
        for i in range(len(nums)):
            if left_sum == right_sums[i + 1]: return i
            left_sum += nums[i]
        return -1
        
        110. Balanced Binary Tree
    def isBalanced(root):
        def _isBalanced(node):
            if not node: return 0
            left = _isBalanced(node.left)
            if left == -1: return -1
            right = _isBalanced(node.right)
            return -1 if (abs(left - right) > 1 or right == -1) else max(left, right) + 1
        return _isBalanced(root) != -1