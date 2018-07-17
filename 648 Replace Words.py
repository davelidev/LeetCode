class Answer(object):
'''648. Replace Words'''
    def replaceWords(dict, sentence):
        hashed = set()
        for word in dict:
            hashed.add(hash(word))
        res = []
        print hashed
        for word in sentence.split(' '):
            replaced_word = word
            for i in range(1, len(word)):
                if hash(word[:i]) in hashed:
                    replaced_word = word[:i]
                    break
            res.append(replaced_word)
        return ' '.join(res)