class Answer(object):'''40. Combination Sum II'''
    def combinationSum2(candidates, target):
        from collections import Counter
        counts = Counter(candidates)
        uniq_nums = counts.keys()
        cur, res = [], []
        def _combinationSum2(target, idx=0):
            if idx >= len(uniq_nums) or target < 0: return
            if target == 0: res.append(cur[:])
            else:
                if counts[uniq_nums[idx]]:
                    counts[uniq_nums[idx]] -= 1
                    cur.append(uniq_nums[idx])
                    _combinationSum2(target - uniq_nums[idx] , idx)
                    cur.pop()
                    counts[uniq_nums[idx]] += 1
                _combinationSum2(target, idx + 1)
        _combinationSum2(target)
        return res