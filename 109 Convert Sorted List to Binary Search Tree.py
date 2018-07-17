class Answer(object):
'''109. Convert Sorted List to Binary Search Tree'''
    def sortedListToBST(head):
        lst = []
        cur = head
        while cur:
            lst.append(cur.val)
            cur = cur.next
        def _sortedListToBST(start, end):
            if start >= end:
                return
            mid = (end + start) / 2
            node = TreeNode(lst[mid])
            node.left = _sortedListToBST(start, mid)
            node.right = _sortedListToBST(mid + 1, end)
            return node
        return _sortedListToBST(0, len(lst))