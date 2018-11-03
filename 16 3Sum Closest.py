class Answer(object):'''16. 3Sum Closest'''
    def threeSumClosest(nums, target):
        min_diff = float('inf')
        min_num = None
        nums.sort()
        for i in range(0, len(nums) - 2):
            j, k = i + 1, len(nums) - 1
            while j < k:
                i_j_k_sum = nums[i] + nums[j] + nums[k]
                diff = abs(i_j_k_sum - target)
                if min_diff > diff:
                    min_diff = diff
                    min_sum = i_j_k_sum
                if i_j_k_sum > target: k -= 1
                else: j += 1
        return min_sum