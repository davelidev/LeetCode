class Answer(object):
'''725. Split Linked List in Parts'''
    def splitListToParts(root, k):
        cur = root
        count = 0
        while cur:
            count += 1
            cur = cur.next
        remainder, elem_per_part = count % k, count / k
        res = []
        cur = root
        while len(res) != k:
            prev = dummy = ListNode('dummy')
            dummy.next = cur
            for i in range(elem_per_part + (0 if not remainder else 1)):
                prev, cur = cur, cur.next
            prev.next = None
            res.append(dummy.next)
            remainder = max(remainder - 1, 0)
        return res