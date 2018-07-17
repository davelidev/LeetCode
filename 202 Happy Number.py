class Answer(object):
'''202. Happy Number'''
    def isHappy(n):
        visited = set()
        while n != 1:
            if n in visited: return False
            visited.add(n)
            x, n = n, 0
            while x:
                n += (x % 10) ** 2
                x /= 10
        return True