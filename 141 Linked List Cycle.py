class Answer(object):
'''141. Linked List Cycle'''
    def hasCycle(head):
        visited = set()
        cur = head
        while cur:
            if cur in visited: return True
            visited.add(cur)
            cur = cur.next
        return False