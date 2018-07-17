class Answer(object):
'''203. Remove Linked List Elements'''
    def removeElements(head, val):
        res = res_cur = ListNode('dummy')
        cur = head
        while cur:
            if cur.val == val:
                cur = cur.next
            else:
                res_cur.next = cur
                cur = cur.next
                res_cur = res_cur.next
                res_cur.next = None
        return res.next