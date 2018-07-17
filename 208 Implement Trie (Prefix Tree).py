class Answer(object):
'''208. Implement Trie (Prefix Tree)'''
    class Trie(object):

        def __init__(self): self.trie_tree = {}

        def insert(self, word):
            cur = self.trie_tree
            for char in word: cur = cur.setdefault(char, {})
            cur[True] = True
        def search(self, word):
            cur = self.trie_tree
            for char in word: 
                if char not in cur: return False
                cur = cur[char]
            return True in cur
        def startsWith(self, prefix):
            cur = self.trie_tree
            for char in prefix:
                if char not in cur: return False
                cur = cur[char]
            return True