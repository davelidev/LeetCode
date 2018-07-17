class Answer(object):
'''290. Word Pattern'''
    def wordPattern(pattern, str):
        pat_to_word = {}
        word_to_pat = {}
        words = str.split(' ')
        if len(words) != len(pattern): return False
        for i, word in enumerate(words):
            if pattern[i] in pat_to_word and pat_to_word[pattern[i]] != word or                 word in word_to_pat and word_to_pat[word] != pattern[i]: return False
            pat_to_word.setdefault(pattern[i], word)
            word_to_pat.setdefault(word, pattern[i])
        return True