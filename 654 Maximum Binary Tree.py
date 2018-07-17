class Answer(object):
'''654. Maximum Binary Tree'''
    def constructMaximumBinaryTree(nums):
        def _constructMaximumBinaryTree(i, j):
            if i == j:
                return
            max_idx, max_val = -1, float('-inf')
            for k in range(i, j):
                if max_val < nums[k]:
                    max_idx, max_val = k, nums[k]
            node = TreeNode(nums[max_idx])
            node.left = _constructMaximumBinaryTree(i, max_idx)
            node.right = _constructMaximumBinaryTree(max_idx + 1, j)
            return node
        return _constructMaximumBinaryTree(0, len(nums))
    def constructMaximumBinaryTree(nums):
        stack = []
        for num in nums:
            node = TreeNode(num)
            while stack and stack[-1].val < num:
                node.left = stack.pop()
            if stack:
                stack[-1].right = node
            stack.append(node)
        return stack[0]