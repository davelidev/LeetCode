class Answer(object):
'''567. Permutation in String'''
    def checkInclusion(s1, s2):
        from collections import Counter
        counts = Counter(s1)
        for i, new_char in enumerate(s2):
            counts[new_char] = counts.get(new_char, 0) - 1
            if not counts[new_char]: del counts[new_char]
            if len(s1) <= i:
                ord_char = s2[i - len(s1)]
                counts[ord_char] = counts.get(ord_char, 0) + 1
                if not counts[ord_char]: del counts[ord_char]
            if not counts: return True
        return False