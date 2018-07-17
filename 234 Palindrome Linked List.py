class Answer(object):
'''234. Palindrome Linked List'''
    def isPalindrome(self, head):
        if not head or not head.next: return True
        num_elem = 0
        cur = head
        while cur:
            num_elem += 1
            cur = cur.next
        prev, cur = None, head
        mid = num_elem / 2
        for _ in range(mid):
            next_node = cur.next
            cur.next = prev
            prev, cur = cur, next_node
        if num_elem & 1: cur = cur.next
        left, right = prev, cur
        while left or right:
            if left.val != right.val: return False
            left, right = left.next, right.next
        return True