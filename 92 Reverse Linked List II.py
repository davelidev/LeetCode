class Answer(object):
'''92. Reverse Linked List II'''
    def reverseBetween(head, m, n):
        dummy = cur = ListNode(0)
        cur.next = head
        for i in range(m): prev, cur = cur, cur.next
        tail1, tail2 = prev, cur
        prev = None
        for i in range(n - m + 1):
            cur.next, prev, cur = prev, cur, cur.next
        tail1.next, tail2.next = prev, cur
        return dummy.next