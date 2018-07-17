class Answer(object):
'''206. Reverse Linked List'''
    def reverseList(head):
        cur = head
        prev = None
        while cur:
            next_node = cur.next
            cur.next = prev
            prev, cur = cur, next_node
        return prev