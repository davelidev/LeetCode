class Answer(object):
'''147. Insertion Sort List'''
    def insertionSortList(head):
        dummy = ListNode(-1000000)
        dummy.next = head
        boundary_prev, boundary = dummy, head
        while boundary:
            node = boundary
            boundary = boundary.next
            boundary_prev.next = boundary
            prev, cur, node.next = dummy, dummy.next, None
            while cur and cur != boundary and cur.val < node.val: prev, cur = cur, cur.next
            tmp = prev.next
            prev.next = node
            node.next = tmp
            if boundary_prev.next != boundary: boundary_prev = boundary_prev.next
        return dummy.next