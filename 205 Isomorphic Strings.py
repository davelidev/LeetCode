class Answer(object):
'''205. Isomorphic Strings'''
    def isIsomorphic(s, t):
        s_to_t = {}
        for i in range(len(s)):
            s_to_t[ord(s[i])] = ord(t[i])
            s_to_t[ord(t[i]) << 10] = ord(s[i])
        for i in range(len(s)):
            if not(s_to_t[ord(s[i])] == ord(t[i]) and s_to_t[ord(t[i]) << 10] == ord(s[i])): return False
        return True