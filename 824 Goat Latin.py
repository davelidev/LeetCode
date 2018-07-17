class Answer(object):
'''824. Goat Latin'''
    def toGoatLatin(S):
        def parse_prefix(word):
            return word if (word[0].lower() in 'aeiou') else (word[1:] + word[0])
        return ' '.join([word + 'ma' + ((i + 1) * 'a')
                         for i, word in enumerate(map(parse_prefix, S.split()))])