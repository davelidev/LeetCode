class Answer(object):
'''82. Remove Duplicates from Sorted List II'''
    def deleteDuplicates(head):
        prev = dummy = ListNode('dummy')
        cur = dummy.next = head
        while cur and cur.next:
            if cur.val == cur.next.val:
                next_diff = cur
                while next_diff.next and next_diff.val == next_diff.next.val:
                    next_diff = next_diff.next
                prev.next = cur = next_diff.next
            else:
                prev, cur = cur, cur.next
        return dummy.next