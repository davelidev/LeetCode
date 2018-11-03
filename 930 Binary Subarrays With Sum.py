class Answer(object):'''930. Binary Subarrays With Sum'''
    def numSubarraysWithSum(A, S):
        sum_to_idx = [A[0]]
        for i in range(1, len(A)):
            sum_to_idx.append(sum_to_idx[-1] + A[i])
        res = sum_so_far = 0
        seen_count = {}
        for num in A:
            seen_count[sum_so_far] = seen_count.get(sum_so_far, 0) + 1
            res += seen_count.get((sum_so_far + num) - S, 0)
            sum_so_far += num
        return res