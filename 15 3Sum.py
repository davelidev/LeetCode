class Answer(object):'''15. 3Sum'''
    def threeSum(nums):
        nums.sort()
        res = []
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            i_num = nums[i]
            j = i + 1
            k = len(nums) - 1
            while j < k:
                j_num = nums[j]
                k_num = nums[k]
                total = i_num + j_num + k_num
                if total == 0:
                    res.append([i_num, j_num, k_num])
                    while j < k and nums[j] == nums[j + 1]:
                        j += 1
                    while j < k and nums[k] == nums[k - 1]:
                        k -= 1
                    j += 1
                elif total > 0:
                    k -= 1
                elif total < 0:
                    j += 1
        return res