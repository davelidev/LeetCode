class Answer(object):
'''523. Continuous Subarray Sum'''
    sum_so_far = 0
    prev_sums = set()
    for i in range(len(lst) - 1, -1, -1):
        sum_so_far += lst[i]
        if (k - (sum_so_far % k)) in prev_sums:
            return True
        prev_sums.add(sum_so_far % k)
    return False