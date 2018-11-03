class Answer(object):'''916. Word Subsets'''
    def wordSubsets(A, B):
        from collections import Counter        
        max_c = {}
        for w2 in B:
            for char, count in Counter(w2).iteritems():
                max_c[char] = max(max_c.get(char, 0), count)

        res = []
        for w1 in A:
            c_word1 = Counter(w1)
            if all(char in c_word1 and c_word1[char] >= max_c[char] for char in max_c):
                res.append(w1)
        return res