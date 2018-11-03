class Answer(object):'''817. Linked List Components'''
    def numComponents(head, G):
        G = set(G)
        cur = head
        prev_in_G = False
        total = 0
        while cur:
            if not prev_in_G and cur.val in G:
                total += 1
            prev_in_G = cur.val in G
            cur = cur.next
        return total