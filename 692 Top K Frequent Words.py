class Answer(object):
'''692. Top K Frequent Words'''
    def topKFrequent(words, k):
        word_to_freq = {}
        for word in words:
            word_to_freq.setdefault(word, 0)
            word_to_freq[word] += 1
        def cmp_func(freq_word1, freq_word2):
            if freq_word1[0] > freq_word2[0] or freq_word1[0] == freq_word2[0] and freq_word1[1] < freq_word2[1]:
                return -1
            return 1
        freq = [(freq, word) for word, freq in word_to_freq.iteritems()]
        freq.sort(cmp_func)
        return [item[1] for item in freq[:k]]