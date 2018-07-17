class Answer(object):
'''76. Minimum Window Substring'''
    def minWindow(s, t):
        from collections import Counter
        start = 0
        t_counts = Counter(t)
        missing_count = len(t_counts)
        i, j = 0, float('inf')
        for end, c in enumerate(s):
            t_counts[c] -= 1
            if t_counts[c] == 0: missing_count -= 1
            while missing_count == 0:
                if end - start < j - i:
                    i, j = start, end + 1
                t_counts[s[start]] += 1
                if t_counts[s[start]] == 1: missing_count += 1
                start += 1
        return s[i: j] if j != float('inf') else ''