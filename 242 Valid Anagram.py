class Answer(object):
'''242. Valid Anagram'''
    def isAnagram(self, s, t):
        from collections import Counter
        return Counter(s) == Counter(t)

    def isAnagram(s, t):
        alphabet_count = 26
        counts = [0] * alphabet_count
        for char in s: counts[ord(char) % alphabet_count] += 1
        for char in t: counts[ord(char) % alphabet_count] -= 1
        return not any(counts)