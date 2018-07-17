class Answer(object):
'''423. Reconstruct Original Digits from English'''
    def originalDigits(self, s):
        '''
        zero: Only digit with z
        two: Only digit with w
        four: Only digit with u
        six: Only digit with x
        eight: Only digit with g
        '''
        from collections import Counter
        s = Counter(s)
        idx_to_word = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        def set_count(char, num, counts):
            counts[num] = s.get(char, 0) - sum(counts[i] for i, word in enumerate(idx_to_word) if char in word and i != num)
        counts = [
            s.get('z', 0), None,
            s.get('w', 0), None,
            s.get('u', 0), None,
            s.get('x', 0), None,
            s.get('g', 0), None,
        ]
        set_count('o', 1, counts)
        set_count('t', 3, counts)
        set_count('f', 5, counts)
        set_count('s', 7, counts)
        set_count('i', 9, counts)
        return ''.join(str(i) * occ for i, occ in enumerate(counts))
            
            583. Delete Operation for Two Strings
    def minDistance(word1, word2):
        if not word1 or not word2: return len(word1) or len(word2)
        dp = [[0] * (len(word2) + 1) for _ in range(len(word1) + 1)]
        for i in range(len(word1)):
            for j in range(len(word2)):
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j], dp[i][j] + (1 if word1[i] == word2[j] else 0))
        return (len(word1) + len(word2) - dp[-1][-1] * 2)