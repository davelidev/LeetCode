class Answer(object):
'''720. Longest Word in Dictionary'''
    def longestWord(words):
        words_by_len = [set([''])]
        for word in words:
            while len(word) >= len(words_by_len): words_by_len.append(set())
            words_by_len[len(word)].add(word)

        for i in range(1, len(words_by_len)):
            prev = words_by_len[i - 1]
            cur = words_by_len[i]
            for word in set(words_by_len[i]):
                if word[:-1] not in prev:
                    cur.remove(word)
            if not cur: return min(prev)
        return min(cur)