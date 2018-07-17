class Answer(object):
'''804. Unique Morse Code Words'''
    def uniqueMorseRepresentations(words):
        mapping = [".-","-...","-.-.","-..",".","..-.","--.",
                   "....","..",".---","-.-",".-..","--","-.",
                   "---",".--.","--.-",".-.","...","-","..-",
                   "...-",".--","-..-","-.--","--.."]
        def covert_to_morse(word):
            return ''.join(mapping[ord(char) - ord('a')] for char in word.lower())
        return len(set(map(covert_to_morse, words)))