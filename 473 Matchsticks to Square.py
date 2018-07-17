class Answer(object):
'''473. Matchsticks to Square'''
    def makesquare(nums):
        sum_of_elems = sum(nums)
        if len(nums) < 4 or sum_of_elems % 4: return False
        nums.sort(reverse=True)
        def _makesquare(pos, sums):
            if pos >= len(nums): return not any(sums)
            next_elem = nums[pos]
            visited = set()
            for i in range(len(sums)):
                if sums[i] - next_elem >= 0 and sums[i] not in visited:
                    sums[i] -= next_elem
                    if _makesquare(pos + 1, sums): return True
                    sums[i] += next_elem
                    visited.add(sums[i])
            return False
        return _makesquare(0, [sum_of_elems / 4 for _ in range(4)])