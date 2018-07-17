class Answer(object):
'''438. Find All Anagrams in a String'''
    def findAnagrams(s, p):
        from collections import Counter
        p_occur = Counter(p)
        s_part_occur = Counter(s[:len(p)])
        res = []
        for i in range(len(s) - len(p)):
            if p_occur == s_part_occur:
                res.append(i)
            s_part_occur[s[i]] -= 1
            if not s_part_occur[s[i]]: del s_part_occur[s[i]]
            s_part_occur.setdefault(s[i + len(p)], 0)
            s_part_occur[s[i + len(p)]] += 1
        if p_occur == s_part_occur: res.append(len(s) - len(p))
        return res