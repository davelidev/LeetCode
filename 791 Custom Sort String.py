class Answer(object):
'''791. Custom Sort String'''
    def customSortString(S, T):
        from collections import Counter
        counts = Counter(T)
        seq = S + ''.join(set(T)-set(S))
        return ''.join(char * counts[char] for char in seq)