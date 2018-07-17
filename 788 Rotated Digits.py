class Answer(object):
'''788. Rotated Digits'''
    def rotatedDigits(N):
        from collections import defaultdict
        dig_map = [0, 1, 5, -1, -1, 2, 9, -1, 8, 6]
        dp = [1, 1, 2, 0, 0, 2, 2, 0, 1, 2] #0: can't flip, 1: same, 2: diff
        for i in range(10, N + 1):
            last_dig = i % 10
            first_part = i / 10
            if dp[last_dig] == 2 and dp[first_part] or dp[first_part] == 2 and dp[last_dig]:
                dp.append(2)
            elif dp[last_dig] == dp[first_part] == 1:
                dp.append(1)
            else:
                dp.append(0)
        return sum(i == 2 for i in dp[1:N+1])