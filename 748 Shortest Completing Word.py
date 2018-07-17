class Answer(object):
'''748. Shortest Completing Word'''
    def shortestCompletingWord(licensePlate, words):
        from collections import Counter
        counts = Counter(char.lower() for char in licensePlate if char.isalpha())
        shortest_word = shortest_len = None
        for word in words:
            word_char_counts = Counter(word.lower())
            if (shortest_len is None or len(word) < shortest_len) and                 all(count <= word_char_counts[plate_c] for plate_c, count in counts.iteritems()):
                shortest_word, shortest_len = word, len(word)
        return shortest_word