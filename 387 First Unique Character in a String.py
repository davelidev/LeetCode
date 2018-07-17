class Answer(object):
'''387. First Unique Character in a String'''
    def firstUniqChar(s):
        from collections import Counter
        counts = Counter(s)
        for i, char in enumerate(s):
            if counts[char] == 1:
                return i
        return -1