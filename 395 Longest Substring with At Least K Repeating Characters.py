class Answer(object):
'''395. Longest Substring with At Least K Repeating Characters'''
    def longestSubstring(s, k):
        def _longestSubstring(s, k):
            letter_count = {}
            for char in s:
                letter_count.setdefault(char, 0)
                letter_count[char] += 1
            split_indicies = [-1]
            for i, char in enumerate(s):
                if letter_count[char] < k:
                    split_indicies.append(i)
            split_indicies.append(len(s))
            if len(split_indicies) != 2:
                max_len = 0
                for i in range(1, len(split_indicies)):
                    idx = split_indicies[i]
                    prev_idx = split_indicies[i - 1]
                    max_len = max (max_len, _longestSubstring(s[prev_idx + 1: idx], k))
                return max_len
            else:
                return len(s)
        return _longestSubstring(s, k)