class Answer(object):
'''767. Reorganize String'''
    def reorganizeString(S):
        from collections import Counter
        counts = [(count, char) for char, count in Counter(S).iteritems()]
        max_freq, max_freq_char = max(counts)
        if max_freq > ((len(S) + 1)/ 2): return ""
        
        res = [[max_freq_char] for _ in range(max_freq)]
        i = 0
        while counts:
            count, char = counts.pop()
            if char != max_freq_char:
                for j in range(i, i + count):
                    res[j % max_freq].append(char)
                i += count
        return ''.join([''.join(x) for x in res])