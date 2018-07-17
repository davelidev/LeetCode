class Answer(object):
'''160. Intersection of Two Linked Lists'''
    def getIntersectionNode(headA, headB):
        cur_a, cur_b = headA, headB
        while cur_a and cur_b: cur_a, cur_b = cur_a.next, cur_b.next
        longer, shorter = (headA, headB) if cur_a else (headB, headA)
        cur = cur_a or cur_b
        while cur: longer, cur = longer.next, cur.next
        while longer != shorter: longer, shorter = longer.next, shorter.next
        return shorter