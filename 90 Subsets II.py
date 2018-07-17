class Answer(object):
'''90. Subsets II'''
    def subsetsWithDup(nums):
        from collections import Counter
        counts = Counter(nums)
        uniq = counts.keys()
        cur, res = [], [[]]
        def _subsetsWithDup(idx=0):
            if idx >= len(uniq): return
            if counts[uniq[idx]]:
                counts[uniq[idx]] -= 1
                cur.append(uniq[idx])
                res.append(cur[:])
                _subsetsWithDup(idx)
                cur.pop()
                counts[uniq[idx]] += 1
            _subsetsWithDup(idx + 1)
        _subsetsWithDup()
        return res