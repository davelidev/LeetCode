class Answer(object):
'''212. Word Search II'''
    
    def findWords(board, words):
        '''build a trie-tree and use back tracing that starts from each coordinate 
        build the trie tree using dictionary, for words abc, adc, abd, it will look like this
        {'a':{'b':{'c':{'word':None},'d':{'word':None}},'d':{'c':{'word':None}}}}'''
        m, n = len(board), len(board[0])
        tree = {}
        for word in words:
            cur = tree
            for char in word:
                cur = cur.setdefault(char, {})
            cur[True] = True
        seen = set()
        word = []
        def dfs(i, j, cur):
            word.append(board[i][j])
            if board[i][j] in cur:
                if True in cur[board[i][j]]: seen.add(''.join(word))
                tmp, board[i][j] = board[i][j], None

                for x, y in [(i + dx, j + dy)
                             for dx, dy in zip([1,0,-1,0], [0,1,0,-1])
                             if 0 <= i + dx < m and 0 <= j + dy < n]:
                    if board[x][y]:
                        dfs(x, y, cur[tmp])
                board[i][j] = tmp
            word.pop()
        for i in range(m):
            for j in range(n):
                dfs(i, j, tree)
        return list(seen)