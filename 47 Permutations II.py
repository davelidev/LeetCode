class Answer(object):'''47. Permutations II'''
    def permuteUnique(nums):
        from collections import Counter
        counts = Counter(nums)
        res = []
        keys = counts.keys()
        cur_perm = []
        def _permuteUnique():
            all_used = True
            for num in keys:
                if num in counts and counts[num]:
                    all_used = False
                    counts[num] -= 1
                    cur_perm.append(num)
                    _permuteUnique()
                    cur_perm.pop()
                    counts[num] += 1
            if all_used:
                res.append(cur_perm[:])
        _permuteUnique()
        return res