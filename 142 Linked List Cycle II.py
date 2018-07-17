class Answer(object):
'''142. Linked List Cycle II'''
    def detectCycle(head):
        if not head or not head.next: return
        slow, fast = head.next, head.next.next
        while (fast and fast.next) and slow != fast:
            slow = slow.next
            fast = fast.next.next
        if not fast or not fast.next: return
        cur = head
        in_loop_cur = fast
        while cur != in_loop_cur:
            cur = cur.next
            in_loop_cur = in_loop_cur.next
        return cur