class Answer(object):
'''241. Different Ways to Add Parentheses'''
    import re
    def diffWaysToCompute(input_vals):
        list_sep_vals = []
        buf = []
        list_sep_vals = re.split('([^\d])', input_vals)
        res = []
        ops = { '+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.div }
        for i in range(0, len(list_sep_vals), 2):
            list_sep_vals[i] = int(list_sep_vals[i])
        dp = {}
        def _diffWaysToCompute(start, end):
            print start, end
            if start == end - 1:
                return [list_sep_vals[start]]
            key = '%d_%d'%(start, end)
            if key in dp:
                return dp[key]
            dp[key] = []
            for i in range(start + 1, end, 2):
                left_combo = _diffWaysToCompute(start, i)
                right_combo = _diffWaysToCompute(i + 1, end)
                for l in left_combo:
                    for r in right_combo:
                        dp[key].append(ops[list_sep_vals[i]](l, r))
            return dp[key]
        return _diffWaysToCompute(0, len(list_sep_vals))