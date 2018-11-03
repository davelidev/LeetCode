class Answer(object):'''19. Remove Nth Node From End of List'''
    def removeNthFromEnd(head, n):
        nth = head
        for _ in range(n): nth = nth.next
        nth, cur, prev = head, nth, None
        while cur: cur, prev, nth = cur.next, nth, nth.next
        if nth == head: return head.next
        if prev and prev.next: prev.next = prev.next.next
        return head