class Answer(object):
'''451. Sort Characters By Frequency'''
    def frequencySort(s):
        from collections import Counter
        counts = Counter(s)
        counts = [(freq, char) for char, freq in counts.iteritems()]
        counts.sort(reverse=True)
        for i in range(len(counts)): counts[i] = counts[i][1] * counts[i][0]
        return ''.join(counts)