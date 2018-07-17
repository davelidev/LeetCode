class Answer(object):
'''763. Partition Labels'''
    def partitionLabels(S):
        from collections import Counter
        counts = Counter(S)
        i = j = 0
        res = []
        while i < len(S):
            seen = set([S[j]])
            while j < len(S) and seen:
                char = S[j]
                seen.add(char)
                counts[char] -= 1
                if not counts[char]: seen.remove(char)
                j += 1
            res.append(j - i)
            i = j
        return res