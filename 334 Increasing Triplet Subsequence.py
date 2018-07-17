class Answer(object):
'''334. Increasing Triplet Subsequence'''
    def increasingTriplet(nums):
        first = second = float('inf')
        for n in nums:
            if n <= first: first = n
            elif n <= second: second = n
            else: return True
        return False

    def increasingTriplet(nums):
        max_last_two = []
        cur_max = float('-inf')
        for num in reversed(nums):
            if cur_max > num: return True
            elif not max_last_two or num > max_last_two[0]: max_last_two = [num]
            elif len(max_last_two) == 2 and max_last_two[0] > num > max_last_two[1]: max_last_two[1] = num
            elif len(max_last_two) == 1 and max_last_two[0] > num: max_last_two.append(num)
            if len(max_last_two) == 2: cur_max = max(cur_max, max_last_two[1])
        return False