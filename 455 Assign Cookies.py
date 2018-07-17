class Answer(object):
'''455. Assign Cookies'''
    def findContentChildren(g, s):
        g.sort(reverse=True)
        s.sort(reverse=True)
        count = 0
        while s and g:
            cookie = s.pop()
            if cookie >= g[-1]:
                g.pop()
                count += 1
        return count