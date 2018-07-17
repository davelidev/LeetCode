class Answer(object):
'''368. Largest Divisible Subset'''
    # sort, dp[i] := (size, largest elem) <= max the size by iterating from the beginning if the dp array. given a b c d, if c is divisible by a, and d is divisible by c, then [a c d] will form a subarray since d will be divisible also by a.