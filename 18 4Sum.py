class Answer(object):'''18. 4Sum'''
    def fourSum(nums, target):
        n = len(nums)
        res = set()
        from collections import defaultdict
        sum_to_ind = defaultdict(list)
        for i in range(2, n):
            for j in range(i + 1, n):
                sum_to_ind[(nums[i] + nums[j])].append((i, [nums[i], nums[j]]))
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                pair_1 = [nums[i], nums[j]]
                pair_1_sum = sum(pair_1)
                new_tar = target - pair_1_sum
                if new_tar in sum_to_ind:
                    for idx, pair_2 in reversed(sum_to_ind[new_tar]):
                        if idx <= j: break
                        res.add(tuple(sorted(pair_1 + pair_2)))
        return list(res)