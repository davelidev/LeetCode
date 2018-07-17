class Answer(object):
'''500. Keyboard Row'''
    def findWords(words):
        keyboard = ['qwertyuiop', 'asdfghjkl', 'zxcvbnm']
        res = []
        for word in words:
            char = word[0].lower()
            for row in keyboard:
                if char in row:
                    if all(char.lower() in row for char in word):
                        res.append(word)
                        break
        return res