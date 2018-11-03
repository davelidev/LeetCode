class Answer(object):'''61. Rotate List'''
    def rotateRight(head, k):
        if not head:
            return
        count = 0
        cur = head
        tail = None
        while cur:
            count += 1
            tail, cur = cur, cur.next
        k = k % count
        if not k:
            return head
        cur = head
        for i in range(count - k - 1):
            cur = cur.next
        head, cur.next, tail.next = cur.next, None, head
        return head