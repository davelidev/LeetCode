class Answer(object):
'''148. Sort List'''
    def sortList(head):
        def _merge_sort(head):
            def _get_from_queue(h1, h2):
                if not h1 and not h2: return None, None, None
                elif not h1 or (h1 and h2 and h2.val < h1.val): return h2, h1, h2.next
                elif not h2 or (h1 and h2 and h2.val >= h1.val): return h1, h1.next, h2
            if not head or not head.next: return head
            slow = fast = head
            while fast and fast.next and fast.next.next: slow, fast = slow.next, fast.next.next
            h1, h2, slow.next = head, slow.next, None
            h1 = _merge_sort(h1)
            h2 = _merge_sort(h2)
            cur = dummy = ListNode('dummy')
            while cur:
                cur.next, h1, h2 = _get_from_queue(h1, h2)
                cur = cur.next
            return dummy.next
        return _merge_sort(head)