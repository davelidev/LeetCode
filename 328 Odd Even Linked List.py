class Answer(object):
'''328. Odd Even Linked List'''
    def oddEvenList(head):
        even_odd_head = [ListNode("even"), ListNode("odd")]
        even_odd = even_odd_head[:]
        cur = head
        toggle = 0
        while cur:
            even_odd[toggle].next = ListNode(cur.val)
            even_odd[toggle] = even_odd[toggle].next
            toggle = 1 - toggle
            cur = cur.next
        even_odd[0].next = even_odd_head[1].next
        return even_odd_head[0].next