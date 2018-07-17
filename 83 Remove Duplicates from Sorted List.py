class Answer(object):
'''83. Remove Duplicates from Sorted List'''
    def deleteDuplicates(head):
        cur = head
        res = res_head = ListNode('dummy')
        while cur:
            if cur.val != res.val:
                res.next = cur
                res = res.next
            cur = cur.next
        res.next = None
        return res_head.next