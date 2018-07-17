class Answer(object):
'''677. Map Sum Pairs'''
    class MapSum(object):

        def __init__(self):
            self.trie_tree = {}

        def insert(self, key, val):
            cur = self.trie_tree
            for char in key:
                cur.setdefault(char, {})
                cur = cur[char]
            cur['val'] = val

        def sum(self, prefix):
            res = 0
            cur = self.trie_tree
            for char in prefix:
                if char not in cur:
                    return 0
                cur = cur[char]
            bfs = [cur]
            vals = []
            while bfs:
                vals.extend([item for item in bfs if type(item) == int])
                bfs = [node[char] for node in bfs for char in (node if type(node) != int else [])]
            return sum(vals)