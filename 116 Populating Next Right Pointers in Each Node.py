class Answer(object):
'''116. Populating Next Right Pointers in Each Node'''
    def connect(root):
        if not root:
            return
        lvl = root
        while lvl.left:
            cur = lvl
            while cur:
                cur.left.next = cur.right
                if cur.next:
                    cur.right.next = cur.next.left
                cur = cur.next
            lvl = lvl.left