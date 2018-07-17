class Answer(object):
'''693. Binary Number with Alternating Bits'''
    def hasAlternatingBits(self, n):
        toggle = n & 1
        while n:
            n, is_one = divmod(n, 2)
            if toggle and not is_one or not toggle and is_one: return False
            toggle = not toggle
        return True