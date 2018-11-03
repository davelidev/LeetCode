class Answer(object):'''23. Merge k Sorted Lists'''
    def mergeKLists(lists):
        import heapq
        heap = [(lst.val, lst) for lst in lists if lst]
        heapq.heapify(heap)
        cur = dummy_head = ListNode('dummy')
        while heap:
            elem, lst = heapq.heappop(heap)
            cur.next = lst
            cur, lst = cur.next, lst.next
            if lst: heapq.heappush(heap, (lst.val, lst))
        return dummy_head.next