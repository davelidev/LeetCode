class Answer(object):'''22. Generate Parentheses'''
    def generateParenthesis(n):
        def _generateParenthesis(sofar, open_paren, closed_paren, res = []):
            if open_paren == n == closed_paren:
                res.append(sofar)
            elif open_paren <= n:
                if open_paren < n: _generateParenthesis(sofar + '(', open_paren + 1, closed_paren)
                if open_paren > closed_paren: _generateParenthesis(sofar + ')', open_paren, closed_paren + 1)
            return res
        return _generateParenthesis('', 0, 0)