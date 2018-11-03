class Answer(object):'''926. Flip String to Monotone Increasing'''
    def minFlipsMonoIncr(S):
        if not S:
            return 0
        num_zeros_right = []
        zeros_so_far = 0
        count = 0
        for bit in reversed(S):
            count += 1
            zeros_so_far += (bit == '0')
            num_zeros_right.append(zeros_so_far)
        num_zeros_right = num_zeros_right[::-1]
        num_zeros_right.append(0)

        optimal = S.count('0')
        ones_so_far = 0
        for i, bit in enumerate(S):
            ones_so_far += (bit == '1')
            zeros_to_right = num_zeros_right[i + 1]
            optimal = min(optimal, ones_so_far + zeros_to_right)
        return optimal