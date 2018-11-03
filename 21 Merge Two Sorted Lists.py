class Answer(object):'''21. Merge Two Sorted Lists'''
    def mergeTwoLists(l1, l2):
        dummy = l = ListNode('dummy')
        while l1 or l2:
            if l1 and l2: l1, l2 = (l1, l2) if l1.val < l2.val else (l2, l1)
            else: l1, l2 = l1 or l2, None
            l.next = l1
            l, l1 = l.next, l1.next
        return dummy.next