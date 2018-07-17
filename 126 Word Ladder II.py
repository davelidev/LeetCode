class Answer(object):
'''126. Word Ladder II'''
    def findLadders(beginWord, endWord, wordList):
    '''O(wl + V + E) time, O(wl * V + E) space, where wl is the max word length
        # Use bfs to find the closest path, words with one letter difference form a path
        initialize and construct the edges like the following:
            for words abc, abd => {'_bc':['abc'], 'a_c':['abc'],'ab_':['abc', 'abd'],'_bd':['abd'],'a_d':['abd']}
            where the keys are the words with 1 character hidden, values are list of words that match the pattern.
            The keys are the nodes, values are the edges
        use bfs for seaching from start to end word
        each node is a word, the adj nodes(hide each of the characters in the word to get all the adj words)'''
        from collections import defaultdict
        n = len(beginWord)
        adj_dict = defaultdict(list)
        for word in wordList:
            for i in range(n):
                adj_dict[word[:i] + '?' + word[i + 1:]].append(word)
        bfs = [([beginWord], beginWord)]
        visited = {beginWord}
        while bfs:
            lvl_visited = set()
            bfs = [(path + [adj], adj)
                   for path, cur in bfs
                   for i in range(n)
                   for adj in adj_dict[cur[:i] + '?' + cur[i + 1:]]
                   if adj not in visited and (lvl_visited.add(adj) is None)]
            visited.update(lvl_visited)
            final_path = [path for path, word in bfs if word == endWord]
            if final_path: return final_path
        return []