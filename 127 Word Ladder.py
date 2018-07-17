class Answer(object):
'''127. Word Ladder'''
    def ladderLength(beginWord, endWord, wordList):
        from collections import defaultdict
        n = len(beginWord)
        adj_dict = defaultdict(list)
        for word in wordList:
            for i in range(n):
                adj_dict[word[:i] + '?' + word[i + 1:]].append(word)
        bfs = [(1, beginWord)]
        visited = {beginWord}
        while bfs:
            bfs = [
                (count + 1, adj)
                for count, cur in bfs
                for i in range(n)
                for adj in adj_dict[cur[:i] + '?' + cur[i + 1:]]
                if adj not in visited and (visited.add(adj) is None)
            ]
            final_path = next((count for count, word in bfs if word == endWord), None)
            if final_path: return final_path
        return 0