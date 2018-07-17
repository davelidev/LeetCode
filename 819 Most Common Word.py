class Answer(object):
'''819. Most Common Word'''
    def mostCommonWord(paragraph, banned):
        from collections import Counter
        counts = Counter((''.join([char for char in paragraph if char.isalpha() or char == ' '])).lower().split(' '))
        freqs = sorted([(freq, word) for word, freq in counts.iteritems()], reverse=True)
        return next((word for freq, word in freqs if word not in banned), None)