class Answer(object):
'''424. Longest Repeating Character Replacement'''
    def characterReplacement(s, k):
        counts = {}
        max_len = start_i = 0
        for end_i, end in enumerate(s):
            counts[end] = counts.get(end, 0) + 1
            max_len = max(max_len, counts[end])
            while ((end_i - start_i + 1) - max_len) > k:
                counts[s[start_i]] -= 1
                start_i += 1
        return max_len + min(k, len(s) - max_len)