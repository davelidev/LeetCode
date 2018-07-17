class Answer(object):
'''820. Short Encoding of Words'''
    def minimumLengthEncoding(words):
        words = sorted([word[::-1] for word in words], reverse=True)
        return sum(len(words[i]) + 1
                   for i in range(len(words) -1, -1, -1)
                   if not (i - 1 >= 0 and words[i - 1].startswith(words[i])))

    def minimumLengthEncoding(words):
        words = [word[::-1] for word in words]
        tree = {}
        for word in words:
            cur = tree
            for char in word:
                cur = cur.setdefault(char, {})
        self.res = 0
        def dfs(depth, cur):
            if not cur:
                self.res += depth + 1
                return
            for adj in cur:
                dfs(depth + 1, cur[adj])
        dfs(0, tree)
        return self.res