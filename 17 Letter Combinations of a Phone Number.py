class Answer(object):'''17. Letter Combinations of a Phone Number'''
    def letterCombinations(digits):
        if not digits: return []
        mappings = ['', '', 'abc', 'def', 'ghi', 'jkl', 'mno', 'pqrs', 'tuv', 'wxyz']
        res = ['']
        for digit in digits: res = [item + char for item in res for char in mappings[int(digit)]]
        return res