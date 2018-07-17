class Answer(object):
'''784. Letter Case Permutation'''
    def letterCasePermutation(S):
        res = []
        S = list(S.lower())
        def _letterCasePermutation(i):
            if i == len(S):
                return res.append(''.join(S))
            elif S[i].isalpha():
                S[i] = S[i].upper()
                _letterCasePermutation(i + 1)
                S[i] = S[i].lower()
            _letterCasePermutation(i + 1)
        _letterCasePermutation(0)
        return res