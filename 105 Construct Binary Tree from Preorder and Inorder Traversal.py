class Answer(object):
'''105. Construct Binary Tree from Preorder and Inorder Traversal'''
    def buildTree(preorder, inorder):
        def _buildTree(pre_s, pre_e, in_s, in_e):
            if pre_s >= pre_e or in_s >= in_e:
                return
            root = TreeNode(preorder[pre_s])
            idx = inorder.index(root.val, in_s, in_e)
            left_dist = idx - in_s
            root.left = _buildTree(pre_s + 1, pre_s + 1 + left_dist, in_s, idx)
            right_dist = in_e - idx - 1
            root.right = _buildTree(pre_s + 1 + left_dist, pre_s + 1 + left_dist + right_dist, idx + 1, idx + 1 + right_dist)
            return root
        return _buildTree(0, len(preorder), 0, len(preorder))
        86. Partition List
        Definition for singly-linked list.
    class ListNode(object):
        def __init__(self, x):
            self.val = x
            self.next = None
    def partition(head, x):
        left = ListNode('Dummy')
        right = ListNode('Dummy')
        left_cur, right_cur = left, right
        cur = head
        while cur:
            if cur.val <= x:
                left_cur.next = cur
                left_cur = left_cur.next
            else:
                right_cur.next = cur
                right_cur = right_cur.next
            cur = cur.next
        left_cur.next = right.next
        return left.next