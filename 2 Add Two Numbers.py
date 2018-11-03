class Answer(object):'''2. Add Two Numbers'''
    def addTwoNumbers(l1, l2):
        l1_cur, l2_cur = l1, l2
        cur = res = ListNode('dummy')
        carry = 0
        while l1 or l2 or carry:
            digit_sum = (l1.val if l1 else 0) + (l2.val if l2 else 0) + carry
            carry, digit = divmod(digit_sum, 10)
            cur.next = ListNode(digit)
            cur = cur.next
            if l1: l1 = l1.next
            if l2: l2 = l2.next
        return res.next