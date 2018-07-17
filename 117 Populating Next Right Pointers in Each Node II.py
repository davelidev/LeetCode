class Answer(object):
'''117. Populating Next Right Pointers in Each Node II'''
    def connect(root):
        lvl_head = root
        while lvl_head:
            cur = lvl_head
            next_lvl_head = next_lvl_cur = TreeLinkNode(-1)
            while cur:
                if cur.left:
                    next_lvl_cur.next = cur.left
                    next_lvl_cur = next_lvl_cur.next
                if cur.right:
                    next_lvl_cur.next = cur.right
                    next_lvl_cur = next_lvl_cur.next
                cur = cur.next
            lvl_head = next_lvl_head.next