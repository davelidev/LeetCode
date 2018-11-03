class Answer(object):'''70. Climbing Stairs'''
    def climbStairs(n):
        if n <= 2: return n
        prev_prev, prev = 1, 2
        cur = None
        for i in range(n - 2):
            cur = prev + prev_prev
            prev, prev_prev = cur, prev
        return cur