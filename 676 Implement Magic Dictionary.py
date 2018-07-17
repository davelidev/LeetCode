class Answer(object):
'''676. Implement Magic Dictionary'''
    class MagicDictionary(object):
        def __init__(self):
            self.words = {}
        def buildDict(self, dict):
            for word in dict:
                for i in range(len(word)):
                    self.words.setdefault(word[:i] + '_' + word[i + 1:], set()).add(word[i])
        def search(self, word):
            for i, char in enumerate(word):
                adjs = self.words.get(word[:i] + '_' + word[i + 1:], set())
                if adjs and ((char not in adjs) or len(adjs) >= 2): return True
            return False