class Answer(object):'''28. Implement strStr()'''
    def strStr(haystack, needle):
        for i in range(0, len(haystack) - len(needle) + 1):
            if haystack[i: i + len(needle)] == needle:
                return i
        return -1