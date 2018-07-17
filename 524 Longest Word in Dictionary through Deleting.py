class Answer(object):
'''524. Longest Word in Dictionary through Deleting'''
    def findLongestWord(s, d):
        def subseq_str(s, subseq_s):
            subseq_s = list(subseq_s)[::-1]
            for char in s:
                if char == subseq_s[-1]: subseq_s.pop()
                if not subseq_s: return True
            return False
        d.sort(key=lambda word: (-len(word), word))
        return next((word for word in d if subseq_str(s, word)), "")