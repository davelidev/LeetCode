class Answer(object):
'''445. Add Two Numbers II'''
    def addTwoNumbers(self, l1, l2):
        
        def _get_stack(node):
            stack = []
            while node:
                stack.append(node.val)
                node = node.next
            return stack
        s1 = _get_stack(l1)
        s2 = _get_stack(l2)
        
        carry = 0
        dummy = ListNode('dummy')
        while s1 or s2 or carry:
            cur_val = carry
            if s1:
                cur_val += s1.pop()
            if s2:
                cur_val += s2.pop()
            carry, cur_val = cur_val/10, cur_val%10
            cur_node = ListNode(cur_val)
            cur_node.next, dummy.next = dummy.next, cur_node
        return dummy.next