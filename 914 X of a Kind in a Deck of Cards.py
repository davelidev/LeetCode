class Answer(object):'''914. X of a Kind in a Deck of Cards'''
    def hasGroupsSizeX( deck):
        from collections import Counter
        counts = Counter(deck)
        counts = set(counts.values())
        min_c = min(counts)
        return any( all((c % i == 0) for c in counts) for i in range(2, min_c + 1)) and min_c > 1