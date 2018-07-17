class Answer(object):
'''211. Add and Search Word - Data structure design'''
    class WordDictionary(object):

        def __init__(self):
            self.trie_tree = {}

        def addWord(self, word):
            cur = self.trie_tree
            for char in word:
                cur = cur.setdefault(char, {})
            cur[True] = word

        def search(self, word):
            stack = [self.trie_tree]
            for char in word:
                if not stack: return False
                if char == '.': stack = [cur[cur_char] for cur in stack for cur_char in cur if cur_char != True]
                else: stack = [cur[char] for cur in stack if char in cur]
            return any(cur[True] for cur in stack if True in cur)