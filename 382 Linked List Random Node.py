class Answer(object):
'''382. Linked List Random Node'''
    import random
    class Solution(object):

        def __init__(self, head):
            self.num_elem = 0
            self.head = cur = head
            while cur:
                self.num_elem += 1
                cur = cur.next

        def getRandom(self):
            idx = random.randint(0, self.num_elem - 1)
            cur = self.head
            while cur:
                if idx == 0:
                    return cur.val
                idx -= 1
                cur = cur.next