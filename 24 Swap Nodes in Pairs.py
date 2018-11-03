class Answer(object):'''24. Swap Nodes in Pairs'''
    def swapPairs(self, head):
        dummy_head = ListNode('dummy')
        dummy_head.next = head
        cur = dummy_head
        while cur and cur.next and cur.next.next:
            nodes = [cur, cur.next, cur.next.next, cur.next.next.next]
            nodes[0].next, nodes[1].next, nodes[2].next, cur = nodes[2], nodes[3], nodes[1], nodes[1]
        return dummy_head.next